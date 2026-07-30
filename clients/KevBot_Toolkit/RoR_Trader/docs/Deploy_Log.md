# Deploy Log — `dev` branch / Railway Worker

Running log of pushes to `dev` that trigger a Railway redeploy. Each push
restarts the Worker container, which causes a 1–3 minute gap in bar
recording (warmup re-runs, no live alerts during restart). Use this log
to correlate cache gaps, alert misses, or WebSocket reconnect events
with deploy windows rather than chasing them as data bugs.

Times are recorded in **UTC** (chart timestamps) and **MT** (Kevin's
local). Worker container restart time = ~30s build + ~30–60s warmup =
~1–2 min unavailable per deploy.

## Format

```
- **HH:MM UTC (HH:MM MT)** — `<short_sha>` <commit subject>
  - Service(s) redeployed: Worker / api / frontend / streamlit
  - Observed cache gap: <bar_start times affected, or "none observed">
  - Notes: anything unusual (stuck deploys, reverts, etc.)
```

## 2026-07-30


- **02:00:52 UTC (20:00 MT 07-29)** — `1c8a6354` **Wave-13 train — ONE CAR, merged; deploy CLEAN (7/7); dispatcher V4.22 → V4.23, PUSH AT RUN END** (board #209, R·auto). ✅ **Market CLOSED** — Wednesday 07-29 post-close (RTH ended 20:00Z), live-trade **DORMANT**. Single car: **PR #139 `feat/dispatcher-push-at-run-end-195`** (board #195 — a headless agent's branch is now pushed by the dispatcher at run end, so work can no longer die unpushed in a worktree; 4 commits `59c4e0ce` feature · `3198cbb9` hermetic acceptance suite · `a4688e2a` worktree discovery instead of assumption · `f04d8b50` §4 fresh-worktree regression set, plus M's reconcile merge `d3f227f1`) → **`1c8a6354`**. Whole-wave diff `backup/dev-pre-wave13-0730`..`origin/dev` = **4 files, 996 insertions / 5 deletions**: `tools/team_dispatcher/dispatcher.py` (+323/−5), `src/test_dispatcher_push_at_run_end_195.py` (new, +663), `src/migrations/run_history_pushed_branch.sql` (new, +14), `tools/team_dispatcher/.gitignore` (+1). Backup branch `backup/dev-pre-wave13-0730` pushed at `b7c09e85` and verified on the remote (local == `origin/`) before the merge.
  - ✅ **M's pre-merge reconcile held — R found NO conflict, exactly as the brief predicted.** The brief's abort condition was "if you hit a conflict, dev moved again and M must redo the reconcile". R proved the negative rather than discovering it during the merge: PR #139's head was `d3f227f1`, **the exact SHA in the brief** (dev had not advanced since M's reconcile), and `git merge-base --is-ancestor origin/dev d3f227f1` returned **true** — i.e. `b7c09e85` was already an ancestor of the PR head, making a conflict *structurally impossible*, not merely unlikely. M's two resolutions are the ones on dev: the docstring renumbered to **V4.23**, and `one_pass()`'s claim-time run record **unioned** so it carries BOTH #195's `worktree`/`branch0`/`worktrees0` and #198's `step_sig`.
  - ✅ **The tree that shipped is byte-identical to the tree that was gated.** `d3f227f1^{tree}` == `1c8a6354^{tree}` == `b4092057f1453ed9cfd0ea0adfef60c6553e02a1`. Because dev was already merged into the branch, GitHub's merge commit added no content, so the gates below were run against the *exact* bytes now on dev — the pre-merge/post-merge re-run distinction collapses to a single provably-equivalent run rather than two hopefully-equivalent ones.
  - Service(s) redeployed: api + Worker + frontend + batch-worker + Data Worker + Streamlit + flat-file-cron (dev auto-deploy, watchPatterns=[] → all rebuild on any dev push), 1 push. **shadow-worker NOT redeployed** (off this path). **No RORT_* flag touched** (a git push cannot alter env-var flag config); R holds no `railway` authority by design.
  - ✅ **DEPLOY CLEAN — all 7 services `success` on the final head `1c8a6354`**: frontend 02:01:19Z, **Worker 02:01:23Z**, Data Worker 02:01:28Z, batch-worker 02:01:30Z, flat-file-cron 02:01:30Z, Streamlit 02:01:31Z, api 02:01:31Z. Combined status state = `success`, total 7/7, settled **~39s** after the merge — the fastest of the 07-29/30 run, and unusually the frontend was *first* rather than the laggard. Watched to settlement, **not** stopped on `pending` (the initial read was 7/7 `pending`). **`Worker` is now green six consecutive trains** (Waves 8, 9, 10, 11, 12, 13); Wave 7's `Worker` red stays closed as a Railway flake.
  - Edge probes (HTTP, post-deploy): api `/health` **200** · api `/api/openapi.json` **200** · api `/api/dev-tasks` **401** unauthenticated (router loaded + admin-gated, **not 5xx** → no import/startup crash) · frontend `/admin/tasks` **200** · frontend `/admin/strategy-health` **200**.
  - **Runtime impact on the deployed Railway services is ZERO.** `dispatcher.py` is local tooling that runs on Kevin's machine only, never in a Railway container (its own header says so); `test_dispatcher_push_at_run_end_195.py` is a standalone module nothing imports; the `.sql` file is an already-applied migration artifact, not executed by any service. No api router, no frontend, no engine / data-loading / resample / warmup / cache path. The 7 rebuilds are purely `watchPatterns=[]` fan-out. Hence the heavy `fidelity_parity_suite.py` was correctly not required (path-based gating per its own docstring scope); the hermetic Fidelity Gate / `synthetic-parity` Action is the gate that applies and it was green.
  - **Gates — 8 suites, all exit 0**, run from a **clean detached worktree** cut at the merged tip `d3f227f1` (never the main checkout — a live dispatcher loop runs out of that tree). Every count matches M's review: `src/test_dispatcher_push_at_run_end_195.py` **ALL 97 CHECKS PASSED** · `src/test_dispatcher_ai_eligible_gate_198.py` **ALL PASS (78)** · `src/test_dispatcher_inbound_check_182.py` **18 passed, 0 failed** · `src/test_dispatcher_step_level_182.py` **ALL PASS (21)** · `src/test_dispatcher_loud_skips_171.py` **ALL PASS (15)** · `src/test_mentions_agent_side_143.py` **ALL PASS (39)** · `tools/team_dispatcher/test_dispatcher_reap.py` **ALL PASS** · **`src/test_ai_eligible_198.py` ALL PASS (52)** via `.venv/bin/python` — *not in the brief's list, run anyway because this diff rewrites `one_pass()`, which is the function #198's gate lives in*. GitHub CI on `d3f227f1`: `build` **pass** ×2, `synthetic-parity` **pass** ×2, `Supabase Preview` skipped; `CLEAN`/`MERGEABLE` re-read immediately before merging.
  - ✅ **Migration confirmed applied and INERT — verification only, R ran NO DDL** (the brief explicitly forbade it; the DDL was applied earlier under Kevin's authorization). Read-only prod query: `run_history.pushed_branch` and `run_history.pushed_at` both exist and are selectable, currently `None` on historical rows — correct, since only a V4.23 loop writes them. Nothing became newly dispatchable from this merge.
  - 🔴 **DISPATCHER RESTART OWED TO M — deliberately NOT performed by R** (the brief forbade it, and a release session must not cycle the loop that dispatched it). Loop **PID 142677** is alive and polling but still holds **V4.22 in memory**.
  - ⚠️ **NEW OBSERVABILITY GAP worth fixing — preflight check (5) will now read GREEN while the loop runs stale code.** The main checkout was `git pull`ed to `1c8a6354` at **02:10:03Z** (by M, not R — R only fetches). Loop PID 142677 started **01:38:30Z**, i.e. **31.5 minutes BEFORE** the file was rewritten, so it definitively loaded V4.22. Preflight (5) compares *file-vs-dev*, **not process-vs-file** — so pulling the checkout without restarting silently converts a visible drift into an invisible one. Every prior wave caught this because the pull was still owed; this wave the pull landed first. Suggested fix: have the loop log its version banner on start and have preflight (5) compare the *running* process's version, not the file's.
  - **Three dispatcher branches deliberately left alone**, per the brief: `fix/dispatcher-failed-run-status-197` and `feat/dispatcher-step-tick-202` (M's next two reconciles — both still fork-behind and both edit the same docstring banner), plus `fix/mass-search-heartbeat-31` and `merge/mrs5a-plus-dev-0722`. **No rebase, reset or force-push, this train or otherwise.**
  - **Nothing armed or disarmed.** Armed set observed unchanged at `ai_eligible=true` → tasks **#202, #195, #184**; #195 and #202 are correctly refused as `Staged` by V4.22's terminal-status rail, and #194 was mid-run at dispatch. No `RORT_*` flag was touched — `RORT_T0_HEALTH_SNAPSHOT` (#160), `RORT_HEALTH_HB_SUPERSEDES_SNAPSHOT` (#157), `RORT_MTF_PB_PREV_EPOCH` (#121) and the #125 settle-retrue all left exactly as they were. No `railway` call of any kind.
  - 📋 **THREE Deploy_Log PRs are now outstanding, and their merge ORDER matters.** This entry is the third unmerged log: **#136** (Wave-11) → **#138** (Wave-12) → **this one** (Wave-13). All three insert at the top of the `## 2026-07-30` section, so merging them out of wave order manufactures a conflict; merging them in order is clean for the first and needs a trivial rebase for each subsequent one. Cleanest option for the next train: **collect all three into one branch** rather than merging three PRs. Per the brief this PR is opened and left **UNMERGED**.
  - Observed cache gap: 1 container cycle in the 02:00–02:02Z window (~1–2 min Worker warmup). **Market CLOSED + live-trade DORMANT** → no live money at risk, no live alerts lost, nothing to exclude from today's intraday paired-% read.

- **01:28:06 UTC (19:28:06 MT 07-29)** — `b7c09e85` **Wave-12 train — ONE CAR, merged; deploy CLEAN (7/7); `Staged` IS NOW TERMINAL, so a release can only ship via R's brief** (board #208, R·auto). ✅ **Market CLOSED** — merged Thursday 07-30 at 01:28Z, i.e. ~5h28m after Wednesday 07-29's RTH close (20:00Z), live-trade **DORMANT**. **Single car, deliberately so:** PR #137 `fix/dispatcher-terminal-staged-198` (board #198 step 8; `TERMINAL_STATUSES = ("Done", "Blocked", "Staged")` — **dispatcher V4.21 → V4.22**; one commit `0289b22e`, already rebased onto the then-current dev head `24ee30c6` by the build run) → **`b7c09e85`**. Final dev head **`b7c09e85`**. Whole-wave diff `backup/dev-pre-wave12-0730`..`origin/dev` = **exactly 3 files, 125 insertions / 47 deletions, nothing else** (`tools/team_dispatcher/dispatcher.py`, `src/test_dispatcher_ai_eligible_gate_198.py`, `src/test_ai_eligible_198.py`). Backup branch `backup/dev-pre-wave12-0730` pushed at `24ee30c6` and verified on the remote before the merge. This train exists to **unblock Kevin's AI-toggle test** — he said he would not test the toggle until it landed.
  - ⭐ **What the one car actually buys.** Without it an **armed `Staged` task self-dispatches**, which bypasses the release brief — the artifact that makes a release auditable. `Staged` means *reviewed, brief held, waiting on a release train*, and the actor at that point is **R executing a brief M wrote** (charter §8), not the task's own assignee. Kevin ruled `Staged` terminal on 07-29, **after** V4.21 had already shipped without it, so `dev` carried a window in which the eligibility toggle could arm a Staged row into a self-ship. That window is now closed. The change also **deliberately REVERSES the "#193 bonus" V4.21 advertised** (armed Staged → R·auto builds half the release train for free) — a train-builder design decision made on purpose, recorded in the constant's own comment so nobody later "cleans up" the odd third entry. `run_requested()`'s button refusals already listed `Staged`, so the V4.21 divergence between the two paths closes itself; nothing changed on that path.
  - 📋 **The guard's real subjects already exist on the board, and are safely UNARMED.** Read-only prod check at merge time: **180 tasks, `ai_eligible` distribution `{False: 180}` — zero armed rows.** Two rows are `Staged`: **#202** and **#195**, both assigned `M`, both unarmed. So the fix changes nothing about the board's present behaviour; what it changes is that **arming either of them can no longer produce a self-dispatch**. Nothing became newly dispatchable, and nothing became newly blocked — Todo / Approval / Review / In Progress all still dispatch (explicitly asserted in the suite, no over-reach).
  - Service(s) redeployed: api + Worker + frontend + batch-worker + Data Worker + Streamlit + flat-file-cron (dev auto-deploy, watchPatterns=[] → all rebuild on any dev push), 1 push. **shadow-worker NOT redeployed** (off this path). **No RORT_* flag touched** (a git push cannot alter env-var flag config); R holds no `railway` authority by design.
  - ✅ **DEPLOY CLEAN — all 7 services `success` on the final head `b7c09e85`**: Data Worker 01:28:28Z, **Worker 01:28:28Z**, flat-file-cron 01:28:30Z, Streamlit 01:28:33Z, batch-worker 01:28:35Z, api 01:28:37Z, **frontend 01:29:41Z**. Combined status state = `success`, settled **~95s** after the merge — frontend was the only laggard, exactly as in Wave 10. Watched to settlement across 2 polls (`pending 1/success 6` → `success 7/7`) — **not** stopped on `pending`, which is where Wave 7's R stopped. **`Worker` is now green FIVE consecutive trains** (Wave 8 `4c6cccea`, Wave 9 `ce7264bf`, Wave 10 `71259239`, Wave 11 `24ee30c6`, Wave 12 `b7c09e85`); Wave 7's `Worker` red stays closed out as a settled Railway flake.
  - Edge probes (HTTP, post-deploy): api `/health` **200** · api `/api/openapi.json` **200** · api `/api/dev-tasks` **401** unauthenticated (router loaded + admin-gated, **not 5xx** → deployed without an import/startup crash) · frontend `/admin/tasks` **200** · frontend `/admin/strategy-health` **200**.
  - **Runtime impact on the deployed Railway services is ZERO — and this is a stronger claim than "docs-only".** The one code file in this train, `tools/team_dispatcher/dispatcher.py`, is **local tooling**: it runs as a loop on Kevin's machine (PID 130391, `--live --loop --poll 20`), not in any Railway container. The other two files are test suites. No api router, no frontend, no engine / data-loading / resample / warmup / cache path, no schema, **no DB migration** (`Supabase Preview` skipped, correctly). The 7 containers rebuilt only because dev auto-deploy has `watchPatterns=[]`, so every dev push rebuilds everything regardless of what changed. Hence the heavy `fidelity_parity_suite.py` was correctly NOT required (path-based gating per its own docstring scope).
  - ✅ **Gates — 7 suites, run TWICE: pre-merge against detached `0289b22e`, then again against the MERGED dev tip `b7c09e85`.** Both runs from a clean detached checkout in the R worktree (`/home/kevin/projects/Kevbot-release`), **never the main checkout** — which matters more than usual this train, since the main checkout is 8 commits stale and its `dispatcher.py` is still the V4.21 file the live loop is running. **Identical results both runs, all exit 0**, and every count matches M's review exactly: `src/test_dispatcher_ai_eligible_gate_198.py` **ALL PASS (78 checks)** · `src/test_dispatcher_inbound_check_182.py` **18 passed, 0 failed** · `src/test_dispatcher_step_level_182.py` **ALL PASS (21 checks)** · `src/test_dispatcher_loud_skips_171.py` **ALL PASS (15 checks)** · `src/test_mentions_agent_side_143.py` **ALL PASS (39 checks)** · `tools/team_dispatcher/test_dispatcher_reap.py` **ALL PASS** · plus **`src/test_ai_eligible_198.py` ALL PASS (52 checks)** — *not in the brief's gate list, added by R because the diff touches that file* (walk 5's fixture had to move off `Staged` once the gate started distinguishing `Review` from `Staged`). Its last assertion is the one worth reading: *"the dispatcher refuses it anyway (TERMINAL_STATUSES — the rail that cannot be forgotten)."* GitHub CI on the #137 head: `synthetic-parity` **pass** ×2 (37s / 44s), `build` **pass** (8s), `Supabase Preview` **skipped**; `mergeStateStatus=CLEAN` / `mergeable=MERGEABLE` re-read immediately before merging, not assumed from M's review.
  - **Reproduced the venv caveat, as Wave 10 warned:** `src/test_ai_eligible_198.py` dies under bare `python3` with `ModuleNotFoundError: fastapi`. It was run through the **main checkout's** interpreter (`/home/kevin/projects/Kevbot-Toolkit/clients/KevBot_Toolkit/RoR_Trader/.venv/bin/python`) with cwd inside the release worktree — the venv exists only in the main checkout, so a fresh worktree must borrow it. Worth promoting out of the Deploy_Log into the gate list itself; three consecutive R sessions have now rediscovered it.
  - 🔴 **DISPATCHER RESTART OWED TO M — deliberately NOT performed by R**, per the brief and per charter §8 as merged in Wave 11. `dev` now carries **V4.22**; the running loop (PID 130391, up ~22 min at merge time) is still on **V4.21**, and the file it loads — the main checkout's — is still the V4.21 file. **Until M restarts, `Staged` is terminal on `dev` but NOT in force in the live loop.** Kevin's AI-toggle test is therefore *not yet* meaningfully unblocked: the merge is a precondition, M's restart + preflight **(5)** verification is what actually arms the semantics. A release session must not cycle the loop that dispatched it.
  - ⏳ **`git pull` the main checkout — now WORSE, and it is the same follow-up Wave 11 flagged.** `/home/kevin/projects/Kevbot-Toolkit` sits on `dev` @ `71259239`, i.e. **8 commits behind** `origin/dev` (`b7c09e85`) — it was 6 behind after Wave 11, and this train added 2. Preflight **(1)** reports "behind origin/dev" and **(4)** reports M-lane charter drift until M pulls. **These two follow-ups are coupled and must be done in order: pull first, then restart** — restarting from the stale checkout would relaunch V4.21 and look like a no-op. **R does not pull the main checkout by design — it is M's working tree.**
  - 📋 **The 07-30 section below jumps Wave-10 → Wave-12.** The **Wave-11 entry is still owed** and lives on the unmerged **PR #136** (`docs/deploy-log-wave11-0730`, board #207); it slots in between when that merges. Two Deploy_Log PRs are now open at once (#136 Wave-11, and this Wave-12 branch) — they touch the same file at the same insertion point, so **merge #136 FIRST**, then this one, or expect a trivial-but-real conflict at the top of the section.
  - **The remaining dispatcher cars stay held back:** `feat/dispatcher-push-at-run-end-195` (Staged), `fix/dispatcher-failed-run-status-197`, `feat/dispatcher-step-tick-202` (Staged). All three edit `dispatcher.py`'s module docstring version banner, as did this car — merging more than one per train guarantees the conflict R's hard limit says to abort on rather than resolve. They ship individually, each after M reconciles the banner onto V4.23+.
  - 📋 **Known-and-parked, NOT touched:** preflight check **(3)** flags `merge/mrs5a-plus-dev-0722` and `fix/mass-search-heartbeat-31` as cut from a stale base. **Both EXPECTED, both E's, both explicitly out of scope per the brief.** R did not rebase, reset or force-push anything, this train or otherwise.
  - **No RORT_* flag was armed** — `RORT_T0_HEALTH_SNAPSHOT` (#160), `RORT_HEALTH_HB_SUPERSEDES_SNAPSHOT` (#157), `RORT_MTF_PB_PREV_EPOCH` (#121) and the #125 settle-retrue all left exactly as they were. **No task was armed** (`ai_eligible` 0/180 — Kevin's stamp is the arming gate, and the brief reserved the arming of the test row to him). #166 branch-protection / merge-queue arming still deferred per Kevin's standing instruction.
  - Observed cache gap: **none expected or chased** — market CLOSED, live-trade DORMANT. The single container cycle in the 01:28–01:30Z window falls entirely outside RTH, so there is no intraday paired-% / alert-lag window to exclude from this train.

- **01:16:42 + 01:17:09 + 01:17:31 UTC (19:16/19:17 MT 07-29)** — `e63d2c55` + `1df5e746` + `24ee30c6` **Wave-11 train — three cars, all merged; deploy CLEAN (7/7); the SHIP-IMMEDIATELY CADENCE IS NOW CHARTER LAW** (board #207, R·auto). ✅ **Market CLOSED** — Wednesday 07-29 post-close (RTH ended 20:00Z; these merges landed 01:17Z on 07-30, ~5h15m after the close), live-trade **DORMANT**. **DOCS-ONLY train, deliberately so.** Merge order = brief order: PR #134 `docs/deploy-log-wave10-0730` (the owed Wave-10 Deploy_Log entry; board #205; `Deploy_Log.md` only, +17/−0) → `e63d2c55`, then **PR #132 `docs/shipping-cadence-default`** (**charter §8**: ship-each-reviewed-branch-immediately is now the **DEFAULT** cadence, plus §8 staleness corrections; `Session_Charters.md` only, +32/−3) → `1df5e746`, then PR #135 `docs/dispatch-dashboard-spec-193` (F's build spec for the dispatch dashboard; board #193 Step 2; **new file** `docs/_active/Spec_Dispatch_Dashboard.md`, +291/−0) → **`24ee30c6`**. **Zero file overlap** — three distinct files, exactly as briefed. Final dev head **`24ee30c6`**. Whole-wave diff `backup/dev-pre-wave11-0730`..`origin/dev` = **exactly 3 markdown files, 340 insertions / 3 deletions, nothing else**. Backup branch `backup/dev-pre-wave11-0730` pushed at `71259239` and verified on the remote before the first merge. Kevin was AFK and had explicitly said to ship what was ready meanwhile.
  - ⭐ **#132 is the car that matters beyond bookkeeping — it is a standing instruction, not a record.** It writes into charter §8 that a branch which has passed M review goes to a train **at once**, as its own single car if nothing else is ready, and that accumulating reviewed branches is now the **exception** requiring Kevin to ask for it in the moment. Its stated rationale, now on `dev`: (a) the cost model that justified batching — each dev push restarts the live Worker, 1–3 min bar-recording gap — **does not currently apply**, because Kevin is not trading live and will not until the dev/prod split (board #204); (b) **merging sooner is the conflict FIX, not the conflict risk** — conflicts come from branch AGE, and 07-29 measured exactly that when four branches (#195/#197/#198/#202) queued behind one train had all bumped `dispatcher.py`'s docstring banner, manufacturing a conflict purely by waiting; (c) the named failure mode is **drift back to caution** — M reverted to batching three times on 07-29 after being told twice not to. §8 also gains two corrections R can confirm from its own existence: **R is `headless` in the agents registry, so releases dispatch from the board rather than a paste-block** (this very session was dispatched that way, not pasted), and **R never restarts the dispatcher loop** — that restart and its preflight (5) verification belong to M.
  - Service(s) redeployed: api + Worker + frontend + batch-worker + Data Worker + Streamlit + flat-file-cron (dev auto-deploy, watchPatterns=[] → all rebuild on any dev push), 3 pushes superseding to the final `24ee30c6` build. **shadow-worker NOT redeployed** (off this path). **No RORT_* flag touched** (a git push cannot alter env-var flag config); R holds no `railway` authority by design.
  - ✅ **DEPLOY CLEAN — all 7 services `success` on the final head `24ee30c6`**: flat-file-cron 01:17:53Z, Data Worker 01:17:54Z, Streamlit 01:17:54Z, batch-worker 01:17:54Z, api 01:17:55Z, frontend 01:17:55Z, **Worker 01:17:59Z**. Combined status state = `success`, settled **~28s** after the watch opened — the fastest settle of the 07-29/30 run, consistent with a 3-file markdown diff. Watched to settlement across 2 polls (`pending 3/success 4` → `success 7/7`) — **not** stopped on `pending`, which is where Wave 7's R stopped. **`Worker` is now green FOUR consecutive trains** (Wave 8 `4c6cccea`, Wave 9 `ce7264bf`, Wave 10 `71259239`, Wave 11 `24ee30c6`); Wave 7's `Worker` red stays closed out as a settled Railway flake.
  - Edge probes (HTTP, post-deploy): api `/health` **200** · api `/api/dev-tasks` **401** unauthenticated (router loaded + admin-gated, **not 5xx** → deployed without an import/startup crash, and #198's `ai_eligible` field from Wave 10 still imports clean) · frontend `/admin/tasks` **200** · frontend `/admin/strategy-health` **200**.
  - **Runtime impact on the deployed Railway services is ZERO.** All three cars are markdown under `docs/`. No api router, no frontend, no `tools/` code, no engine / data-loading / resample / warmup / cache path, no schema. Hence the heavy `fidelity_parity_suite.py` was correctly NOT required (path-based gating per its own docstring scope), and the dispatcher suites had nothing to bind to. The hermetic Fidelity Gate / `synthetic-parity` GitHub Action is the gate that applies and it was **green on all three PR heads**, as was `frontend-gate` / `build`; `Supabase Preview` **skipped** on all three (no migration).
  - **Gates (R·auto): none run, and correctly so — markdown-only train, briefed as such.** Pre-merge validation instead, per PR and pulled from GitHub rather than assumed: all three confirmed `MERGEABLE` / `CLEAN`; **one commit each** (`7608a6ae`, `61751718`, `25b0ad94`), matching the brief exactly; CI verified green per-PR (`synthetic-parity` SUCCESS, `build` SUCCESS, `Supabase Preview` SKIPPED); file lists checked against the briefed scope and **zero file overlap** confirmed. **#132 and #135 were each re-confirmed `MERGEABLE`/`CLEAN` after the preceding merge advanced `dev`** — GitHub returned `UNKNOWN`/`UNKNOWN` on both while recomputing mergeability, and R **polled until it resolved rather than merging on a stale read**.
  - 📋 **One clean-base deviation, flagged and harmless.** Board #153 wants each car cut from the then-current dev head. `git merge-base` against the pre-train head `71259239`: #134 = `71259239` ✅ and #135 = `71259239` ✅, but **#132 = `ce7264bf`** — i.e. #132 was cut from the Wave-9 head, one merge (Wave 10) behind. R proceeded because the check's actual purpose passed: #132's diff is the single file `Session_Charters.md` (+32/−3), it merged with **zero conflicts**, and the whole-wave diff confirms nothing unrelated rode along. Nothing was rebased, reset or force-pushed to fix it.
  - **No DB migration in this train** (all three cars are docs; no schema change). **No dispatcher restart owed by this train** (no `dispatcher.py` change) — and per §8 as merged here, R would not perform one regardless.
  - ✅ **Wave-10's owed dispatcher restart is CONFIRMED DONE by M** (verified, not assumed): `dev` carries **V4.21**, the main checkout's `dispatcher.py` reads **V4.21**, and the loop is alive on it (PID 130391, `--live --loop --poll 20`). Preflight check **(5)** is green. So **#198's eligibility gate is now genuinely in force**, not merely on `dev` — Wave 10's one red follow-up is closed.
  - ⏳ **ONE FOLLOW-UP OWED to M — `git pull` the main checkout.** `/home/kevin/projects/Kevbot-Toolkit` sits on `dev` @ `71259239`, i.e. **6 commits / 3 merges behind** `origin/dev` (`24ee30c6`). Until M pulls, preflight **(1)** reports "behind origin/dev" and **(4)** reports M-lane charter drift — *caused by this train*, since #132 changed `Session_Charters.md`, an M-lane doc. **R does not pull the main checkout by design — it is M's working tree.** Note the irony worth not missing: the doc M must pull is the one telling M to stop batching.
  - **The three ready dispatcher cars remain deliberately held back:** `feat/dispatcher-push-at-run-end-195`, `fix/dispatcher-failed-run-status-197`, `feat/dispatcher-step-tick-202`, plus the in-flight Staged patch for #198. **All four edit `dispatcher.py`'s module docstring version banner**, so merging more than one per train guarantees the conflict R's hard limit says to abort on rather than resolve. They ship individually, each after M reconciles the banner. **This train was scoped docs-only precisely so it could not collide with them** — an M·auto session was building the #198 Staged patch concurrently, and this wave touched no file it touches.
  - 📋 **Known-and-parked, NOT touched:** preflight check **(3)** flags `merge/mrs5a-plus-dev-0722` (fork-behind 165) and `fix/mass-search-heartbeat-31` (fork-behind 15) as cut from a stale base. **Both EXPECTED, both E's, both explicitly out of scope per the brief.** R did not rebase, reset or force-push anything, this train or otherwise.
  - **No RORT_* flag was armed** — `RORT_T0_HEALTH_SNAPSHOT` (#160), `RORT_HEALTH_HB_SUPERSEDES_SNAPSHOT` (#157), `RORT_MTF_PB_PREV_EPOCH` (#121) and the #125 settle-retrue all left exactly as they were. **No task was armed** (`ai_eligible` untouched — Kevin's stamp is the arming gate). #166 branch-protection / merge-queue arming still deferred per Kevin's standing instruction.
  - Observed cache gap: **none expected or chased** — market CLOSED, live-trade DORMANT. The 3 container cycles in the 01:16–01:18Z window fall entirely outside RTH, so there is no intraday paired-% / alert-lag window to exclude from this train.

- **00:58:35 + 01:00:48 UTC (18:58/19:00 MT 07-29)** — `097c10f0` + `71259239` **Wave-10 train — two cars, both merged; deploy CLEAN (7/7); the AI-ELIGIBLE TOGGLE (#198) IS LIVE** (board #205, R·auto). ✅ **Market CLOSED** — Wednesday 07-29 post-close (RTH ended 20:00Z), live-trade **DORMANT**. Merge order = brief order, docs first so a failure on the code car would leave the docs landed: PR #131 `docs/deploy-log-wave9-0729` (the owed Wave-9 Deploy_Log entry; board #201; `Deploy_Log.md` only, +12/−0) → `097c10f0`, then **PR #133 `feat/ai-eligible-toggle-198`** — opened by R, the branch had no PR — (board #198: `dev_tasks.ai_eligible` column + API + board UI toggle + **dispatcher V4.19 → V4.21**; 4 commits `68aa2e8e` API/migration · `cd50b2ff` UI · `faf6af59` gate · `83bc5726` terminal-status guard) → **`71259239`**. **Zero file overlap**; final dev head **`71259239`**. Whole-wave diff `backup/dev-pre-wave10-0730`..`origin/dev` = **13 files, 1367 insertions / 56 deletions**. Backup branch `backup/dev-pre-wave10-0730` pushed at `ce7264bf` and verified on the remote before the first merge. **This is the first Wave in the 07-29/30 run with real DEPLOYED runtime impact** — every prior wave today was docs or local-tooling only; #198 ships an api router change *and* frontend board UI.
  - Service(s) redeployed: api + Worker + frontend + batch-worker + Data Worker + Streamlit + flat-file-cron (dev auto-deploy, watchPatterns=[] → all rebuild on any dev push), 2 pushes superseding to the final `71259239` build. **shadow-worker NOT redeployed** (off this path). **No RORT_* flag touched** (a git push cannot alter env-var flag config); R holds no `railway` authority by design.
  - ✅ **DEPLOY CLEAN — all 7 services `success` on the final head `71259239`**: Data Worker 01:01:18Z, **Worker 01:01:19Z**, Streamlit 01:01:21Z, batch-worker 01:01:21Z, flat-file-cron 01:01:21Z, api 01:01:24Z, **frontend 01:02:22Z**. Combined status state = `success`, settled ~90s after the watch opened. Watched to settlement across 4 polls (6 green + frontend `pending` → 7/7 `success`) — **not** stopped on `pending`. **`Worker` is now green three consecutive trains** (Wave 8 `4c6cccea`, Wave 9 `ce7264bf`, Wave 10 `71259239`); Wave 7's `Worker` red is closed out as a Railway flake.
  - ✅ **THE API SIDE IS PROVABLY LIVE, not merely deployed.** The deployed api's own OpenAPI schema at `/api/openapi.json` (**200**) now contains `ai_eligible`, in the `PATCH /api/dev-tasks/{task_id}` description: *"Closing a task DISARMS it (`ai_eligible=false`, board #198)"*. That string exists only in `83bc5726`'s code, so the running api container is serving Wave-10 code — this is a positive fingerprint, not an inference from a green deploy badge.
  - Edge probes (HTTP, post-deploy): api `/health` **200** · api `/api/docs` **200** · api `/api/openapi.json` **200** · api `/api/dev-tasks` **401** unauthenticated (router loaded + admin-gated, **not 5xx** → the new `ai_eligible` field did not crash import/startup) · frontend `/admin/tasks` **200** · frontend `/` **307** (login redirect).
  - ✅ **Migration confirmed applied and INERT — nothing became newly dispatchable.** Read-only prod query: `dev_tasks.ai_eligible` exists and **all 177 rows are `false`** (distribution `{False: 177}`). Per the brief the DDL was already applied under Kevin's authorization, so **R ran no DDL** — this is verification only. The dispatcher cannot pick up any task from this merge until Kevin flips a toggle.
  - ⏳ **Frontend toggle NOT visually confirmed — handed to Kevin/M to eyeball.** The frontend container deployed SUCCESS and `/admin/tasks` serves 200, but R could not fingerprint the UI headless: the literal string `AI-eligible` was absent from all 12 statically-referenced `_next/static/chunks/*.js`. **This probe is blind, not negative** — control strings `Blocked` and `Staged` (board status labels that certainly exist in the shipped board UI) are *also* 0/12, i.e. the board view is a dynamically-imported chunk the initial HTML never names. A real confirmation needs an authenticated admin session (Kevin's JWT) + a browser; the Playwright MCP was not exposed this headless run. Low risk: api fingerprint positive, frontend build SUCCESS, `/admin/tasks` 200, and #198's UI compiled clean through the `frontend-gate` / `build` Action.
  - **Gates — all 7 suites, run TWICE: once pre-merge against a LOCAL merge preview, then again against the MERGED dev tip `71259239`.** Both runs from clean detached worktrees cut from `origin/dev` (never the main checkout). The pre-merge run is the notable one: rather than testing the #198 branch head in isolation, R merged *both* cars into a throwaway worktree first — so the gated tree was byte-equivalent to the post-merge result, and a conflict or cross-car interaction would have surfaced **before** anything touched `dev`. Both merges applied with zero conflicts. Identical results both runs, all exit 0: `src/test_dispatcher_ai_eligible_gate_198.py` **ALL PASS (74 checks)** · `src/test_dispatcher_inbound_check_182.py` **18 passed, 0 failed** · `src/test_dispatcher_step_level_182.py` **ALL PASS (21 checks)** · `src/test_dispatcher_loud_skips_171.py` **ALL PASS (15 checks)** · `src/test_mentions_agent_side_143.py` **ALL PASS (39 checks)** · `tools/team_dispatcher/test_dispatcher_reap.py` **ALL PASS** · `src/test_ai_eligible_198.py` **ALL PASS (52 checks)** via `.venv/bin/python`. GitHub CI on the #133 head: `synthetic-parity` **pass** (37s/39s), `build` **pass** (1m29s), Supabase Preview skipped.
  - 📋 **Two brief-vs-reality notes (flagged, neither fatal).** (1) `test_ai_eligible_198.py` reported **52** checks where the brief predicted **38** — the tip `83bc5726` terminal-status guard added checks after M's count; more coverage, all green, so R proceeded. (2) The brief's warning that this suite dies under bare `python3` (`ModuleNotFoundError: fastapi`) is real and was respected — it was run through `.venv/bin/python`; note the venv lives only in the **main checkout**, so a fresh worktree must borrow that interpreter (`/home/kevin/projects/Kevbot-Toolkit/clients/KevBot_Toolkit/RoR_Trader/.venv/bin/python`) with cwd inside the worktree.
  - 🔴 **DISPATCHER RESTART OWED TO M — deliberately NOT performed by R.** dev now carries **V4.21**; the running loop still holds **V4.19**, so preflight check (5) will flag file-vs-dev drift until M restarts and verifies. A release session must not cycle the loop that dispatched it. **Until M restarts, #198's eligibility gate is on `dev` but not in force** — the live loop is still dispatching on V4.19 semantics.
  - **Three ready dispatcher cars deliberately held back:** `feat/dispatcher-push-at-run-end-195`, `fix/dispatcher-failed-run-status-197`, `feat/dispatcher-step-tick-202`. All three edit `dispatcher.py`'s module docstring version banner, as does #198 — merging more than one in a single train guarantees a docstring conflict, and R's hard limit is to abort on conflict rather than resolve it. **#198 shipped alone so it could not be blocked by another car's conflict.** The other three ship next with the version numbering reconciled deliberately (#202 also still unreviewed).
  - **No RORT_* flag was armed** — `RORT_T0_HEALTH_SNAPSHOT` (#160), `RORT_HEALTH_HB_SUPERSEDES_SNAPSHOT` (#157), `RORT_MTF_PB_PREV_EPOCH` (#121) and the #125 settle-retrue all left exactly as they were. **No task was armed** (`ai_eligible` stays 0/177 — Kevin's stamp is the arming gate). #166 branch-protection / merge-queue arming still deferred per Kevin's standing instruction.
  - Observed cache gap: **none expected or chased** — market CLOSED, live-trade DORMANT. The 2 container cycles in the 00:58–01:03Z window fall entirely outside RTH, so there is no intraday paired-% / alert-lag window to exclude from this train.

## 2026-07-29


- **23:16:59 + 23:17:48 UTC (17:16/17:17 MT)** — `ea0f8dcd` + `ce7264bf` **Wave-9 train — two cars, both merged; deploy CLEAN (7/7); Worker green two trains running** (board #201, R·auto). ✅ **Market CLOSED** — Wednesday 07-29 post-close (RTH ended 20:00Z), live-trade **DORMANT**. **DOCS-ONLY train.** Merge order = brief order: PR #129 `docs/deploy-log-wave8-0729` (the owed Wave-8 Deploy_Log entry; board #196; `Deploy_Log.md` only, +9/−0) → `ea0f8dcd`, then PR #130 `docs/retrofit-rule-182` (**charter §7: process-chain step schema + `mode=discuss` semantics + Kevin's retrofit rule**; board #182 Step 9; `Session_Charters.md` only, +17/−1) → `ce7264bf`. **Zero file overlap**; final dev head **`ce7264bf`**. Whole-wave diff `backup/dev-pre-wave9-0729`..`origin/dev` = **exactly 2 markdown files, 26 insertions / 1 deletion, nothing else**. Backup branch `backup/dev-pre-wave9-0729` pushed at `4c6cccea` and verified on the remote before the first merge. **#195, #198 and `fix/mass-search-heartbeat-31` deliberately NOT in this train** (all three await review — #195/#198 with M, #31 with E).
  - Service(s) redeployed: api + Worker + frontend + batch-worker + Data Worker + Streamlit + flat-file-cron (dev auto-deploy, watchPatterns=[] → all rebuild on any dev push), 2 pushes superseding to the final `ce7264bf` build. **shadow-worker NOT redeployed** (off this path). **No RORT_* flag touched** (a git push cannot alter env-var flag config); R holds no `railway` authority by design.
  - ✅ **DEPLOY CLEAN — all 7 services `success` on the final head `ce7264bf`**: Streamlit 23:18:11Z, **Worker 23:18:14Z**, batch-worker 23:18:17Z, api 23:18:23Z, flat-file-cron 23:18:31Z, frontend 23:19:02Z, Data Worker 23:19:17Z. Combined status state = `success`, settled ~90s after the watch opened. Watched to settlement across 5 polls (`pending 3/success 4` → `pending 1/success 6` → `success 7/7`) — **not** stopped on `pending`, which is where Wave 7's R stopped. **`Worker` is now green two consecutive trains** (Wave 8 `4c6cccea` 19:07:03Z, Wave 9 `ce7264bf` 23:18:14Z), confirming Wave 7's `Worker` red as a settled Railway flake rather than a recurring break. Nothing to escalate.
  - Edge probes (HTTP, post-deploy): api `/health` **200** · api `/api/dev-tasks` **401** unauthenticated (router loaded + admin-gated, **not 5xx** → deployed without an import/startup crash) · frontend `/admin/tasks` **200** · frontend `/admin/strategy-health` **200**.
  - **Runtime impact on the deployed Railway services is ZERO:** both cars are markdown under `docs/`. No api router, no frontend, no `tools/` code, no engine / data-loading / resample / warmup / cache path. Hence the heavy `fidelity_parity_suite.py` was correctly NOT required (path-based gating per its own docstring scope), and the dispatcher suites had nothing to bind to. The hermetic Fidelity Gate / `synthetic-parity` GitHub Action is the gate that applies and it was **green on both PR heads** (`63d02be9`, `6542b991`), as was `build`; `Supabase Preview` skipped on both (no migration). Every PR's file list was pulled from GitHub and checked against that scope rather than assumed.
  - **Gates (R·auto): none run, and correctly so — markdown-only train, briefed as such.** Pre-merge validation instead: both branches confirmed `MERGEABLE`/`CLEAN`; `git merge-base` for **both** heads against `4c6cccea` = `4c6cccea` exactly, i.e. each cut from the then-current dev head with nothing dragged in (board #153 clean-base contract); **one commit each**, matching the brief; CI verified green per-PR via `gh pr checks`; zero file overlap verified from the GitHub file lists. #130 was re-confirmed `MERGEABLE`/`CLEAN` after #129 advanced dev to `ea0f8dcd` (GitHub returned `UNKNOWN` while recomputing — R polled rather than merging on a stale read).
  - **No DB migration in this train** (both cars are docs; no schema change). **No dispatcher restart owed** (no `dispatcher.py` change).
  - Observed cache gap: 2 pushes in the 23:17–23:19Z window each cycle the Worker container (~1–2 min warmup per restart, superseded by Railway to the final `ce7264bf` build). **Market CLOSED + live-trade DORMANT** → no live money at risk, no live alerts lost, nothing to exclude from today's intraday paired-% read.
  - ✅ **Wave-8's two owed follow-ups are CONFIRMED DONE** (verified by R, not assumed): the main checkout was restarted onto **V4.19** (`dispatcher.py` header reads V4.19; loop PID 31591 `--live --loop --poll 20`), and the armed SessionStart hook `.claude/hooks/team_board_context.py` is now the **6,029-byte** copy with **9** `mention` references. Independent proof the delivery path is live end-to-end: **this R session's own dispatch prompt carried an `=== UNSEEN @MENTIONS FOR R ===` block** (E's #116 sign-off) — the V4.19 feature working in production, not just deployed.
  - ⏳ **ONE FOLLOW-UP OWED to M — `git pull` the main checkout.** `/home/kevin/projects/Kevbot-Toolkit` is on `dev` @ `4c6cccea`, i.e. **2 merges behind** `origin/dev` (`ce7264bf`). Until M pulls, preflight check **(1)** reports "behind origin/dev" and check **(4)** reports M-lane charter drift (#130 changed `Session_Charters.md`, an M-lane doc). R does not pull the main checkout by design — it is M's working tree.
  - 📋 **Known-and-parked, NOT touched:** preflight check **(3)** still flags `merge/mrs5a-plus-dev-0722` as cut from a stale base (fork-behind 154). Expected and explicitly out of scope per the brief — parked for E. R did not rebase, reset or force-push anything, this train or otherwise.

- **19:06:21 + 19:06:39 UTC (13:06 MT)** — `999c0393` + `4c6cccea` **Wave-8 train — two cars, both merged; deploy CLEAN (7/7); Wave-7's Worker red HEALED** (board #196, R·auto). ⚠️ **Market OPEN** — Wednesday 07-29, merges landed mid-RTH (13:30–20:00Z); Kevin explicitly approved the midday window before dispatch ("I'm okay pushing midday today… it's no big deal today"). Live-trade **DORMANT**. Merge order = brief order, docs first so a failure on the code car would leave the docs landed: PR #128 `docs/deploy-log-wave7-0729` (the owed Wave-7 Deploy_Log entry; board #191; `Deploy_Log.md` only, +13/−0) → `999c0393`, then PR #127 `feat/mentions-agent-side-143` (**dispatcher V4.18 → V4.19**, agent-side @-mention delivery; board #143 step 3) → `4c6cccea`. **Zero file overlap**; final dev head **`4c6cccea`**. Whole-wave diff `backup/dev-pre-wave8-0729`..`origin/dev` = **5 files, 845 insertions / 4 deletions**: `docs/Deploy_Log.md` (+13), `tools/team_dispatcher/dispatcher.py` (+68/−4), `tools/team_dispatcher/mentions.py` (new, +278), `tools/team_dispatcher/hooks/team_board_context.py` (new, +149), `src/test_mentions_agent_side_143.py` (new, +341). Backup branch `backup/dev-pre-wave8-0729` pushed at `3397e070` before the first merge. **`fix/mass-search-heartbeat-31` deliberately NOT in this train** (pushed but unreviewed; board #31 sits with E).
  - Service(s) redeployed: api + Worker + frontend + batch-worker + Data Worker + Streamlit + flat-file-cron (dev auto-deploy, watchPatterns=[] → all rebuild on any dev push), 2 pushes superseding to the final `4c6cccea` build. **shadow-worker NOT redeployed** (off this path). **No RORT_* flag touched** (a git push cannot alter env-var flag config); R holds no `railway` authority by design.
  - ✅ **DEPLOY CLEAN — all 7 services `success` on the final head `4c6cccea`**: flat-file-cron 19:07:00Z, frontend 19:07:02Z, **Worker 19:07:03Z**, Data Worker 19:07:05Z, batch-worker 19:07:11Z, api 19:07:20Z, Streamlit 19:08:28Z. Combined status state = `success`, settled 52s after the watch opened. The intermediate head `999c0393` was **also 7/7 green** (19:06:46Z–19:06:51Z). Watched to settlement — **not** stopped on `pending`.
  - ✅ **Wave-7's `Railway - Worker` failure is RESOLVED — it was a Railway flake, and this push self-healed it exactly as predicted.** Prior head `3397e070` carried `Railway-RoR Trader - Worker` = **failure** with the other 6 green (see the Wave-7 entry below); on `4c6cccea` the same context reports **success** at 19:07:03Z. This was the explicit abort condition for this train — a *second consecutive* Worker failure would have meant "no longer a flake, hand to M/E". It did not recur, so **no abort, no escalation**. The Wave-7 partial-deploy caveat can be considered closed.
  - **Runtime impact on the deployed Railway services is ZERO.** #128 is markdown. #127 is **dispatcher tooling that runs on Kevin's machine only, never on Railway** (`dispatcher.py` header: "Runs on Kevin's machine only (never Railway)"); its one file under `src/` is a standalone test module that nothing imports. No api router, no frontend, no engine / data-loading / resample / warmup / cache path. Hence the heavy `fidelity_parity_suite.py` was correctly NOT required (path-based gating per its own docstring scope); the hermetic Fidelity Gate / `synthetic-parity` GitHub Action is the gate that applies and it was **green on both PR heads**, as was `frontend-gate` / `build`.
  - **Gates — 5 dispatcher suites, run TWICE: once pre-merge at the #127 branch head `7a76605a`, then again against the MERGED dev tip `4c6cccea`.** Both runs from a clean detached worktree (`/home/kevin/projects/Kevbot-release`), never the main checkout. Identical results both times, all exit 0: `src/test_mentions_agent_side_143.py` **ALL PASS (39 checks)** · `src/test_dispatcher_inbound_check_182.py` **18 passed, 0 failed** · `src/test_dispatcher_step_level_182.py` **ALL PASS (21 checks)** · `src/test_dispatcher_loud_skips_171.py` **ALL PASS (15 checks)** · `tools/team_dispatcher/test_dispatcher_reap.py` **ALL PASS**. The merged-tip re-run is the one that would have caught a silent interaction between the two cars; it found none.
  - **No DB migration in this train** (no schema change; `mentions.py` reads/writes existing board tables).
  - ⏳ **TWO FOLLOW-UPS OWED — deliberately left to M, NOT done by this release session.** (1) **Dispatcher restart owed:** dev now carries **V4.19**; the running loop (PID 12326, `--live --loop --poll 20`) is still on **V4.18**, so preflight check (5) will flag file-vs-dev drift until M restarts and verifies. A release session must not cycle the loop that dispatched it. (2) **SessionStart hook NOT armed:** #127 lands the *tracked canonical* copy at `tools/team_dispatcher/hooks/team_board_context.py` (6,029 bytes, 9 `mention` references); the *armed* copy at `.claude/hooks/team_board_context.py` is still the old 4,509-byte version with **0** `mention` references. Arming is a deliberate `cp` that M performs as a second touch. **Until both are done, @-mention delivery is on dev but not live.**

- **18:36:52 + 18:37:41 UTC (12:36/12:37 MT)** — `02ac05b2` + `3397e070` **Wave-7 train — two cars, both merged; ⚠️ deploy PARTIAL (6/7)** (board #191, R·auto). ⚠️ **Market OPEN** — Wednesday 07-29, merges landed mid-RTH (13:30–20:00Z); Kevin explicitly approved the RTH window and its ~2 container restarts before dispatch. Live-trade **DORMANT**. **DOCS-ONLY train.** Merge order = brief order: PR #125 `docs/deploy-log-wave6-0729` (the owed Wave-6 Deploy_Log entry; board #190; `Deploy_Log.md` only, +10/−0), then PR #126 `docs/m-handoff-0729-v2` (M successor handoff + roster row — `Session_Charters.md` +3/−1 and a new `docs/_active/handoffs/M_handoff_2026-07-29_v2.md` +151/−0; **this is the car that clears preflight check (4) M-lane charter drift**). **Zero file overlap**; final dev head `3397e070`. Whole-wave diff `backup/dev-pre-wave7-0729`..`origin/dev` = **exactly 3 markdown files, 164 insertions / 1 deletion, nothing else**.
  - Service(s) redeployed: api + Worker + frontend + batch-worker + Data Worker + Streamlit + flat-file-cron (dev auto-deploy, watchPatterns=[] → all rebuild on any dev push), 2 pushes superseding to the final `3397e070` build. **shadow-worker NOT redeployed** (off this path). **No RORT_* flag touched** (a git push cannot alter env-var flag config).
  - **Runtime impact on the deployed services is ZERO:** both cars are markdown under `docs/`. No api router, no frontend, no `tools/` code, no engine/data-loading/resample/warmup/cache path. Hence the heavy `fidelity_parity_suite.py` was correctly NOT required (path-based gating per its own docstring scope), and the dispatcher suites had nothing to bind to; the hermetic Fidelity Gate / `synthetic-parity` GitHub Action is the gate that applies and it was **green on both PR heads** (`3a0d4de5`, `8df15ac1`), as was `frontend-gate` / `build`. Every PR's file list was pulled from GitHub and checked against that scope rather than assumed. **No dispatcher restart owed this train** (no `dispatcher.py` change).
  - **No DB migration in this train** (both cars are docs; no schema change).
  - 🔴 **DEPLOY PARTIAL — `Worker` deploy FAILED on the final head; the other 6 are green.** Commit-statuses on `3397e070`: Streamlit **success** 18:38:14Z, api **success** 18:38:17Z, frontend **success** 18:38:27Z, flat-file-cron **success** 18:38:36Z, Data Worker **success** 18:38:38Z, batch-worker **success** 18:39:37Z — and **`Worker` `failure` ("Deployment failed") 18:39:11Z**. Combined status state = `failure`. R watched **25 polls over ~9 minutes** (18:38:29Z → 18:47:21Z); `Worker` never retried and never flipped, so this is a settled failure and **not** a stopped-on-`pending` reading error.
  - ✅ **Blast radius measured, not assumed — live impact NIL.** (1) Railway kept the previous deployment serving: `Worker` deployed **SUCCESS on `02ac05b2`** (the #125 merge) at **18:37:33Z**, 68s before the failed build — so the live Worker is running `02ac05b2`, which differs from dev head by **exactly the two markdown files in #126**, i.e. zero functional delta. (2) The Worker is demonstrably **alive and current**: read-only prod probe at 18:48:22Z showed `live_bars` newest writes **sub-second fresh** (SPY + TSLA @ 18:48:23Z) and `shadow_heartbeats` updating within seconds (strategies 271/269/267 @ 18:48:2xZ). **No bar-recording gap, no stall.**
  - **Read: Railway-side deploy flake, not a code break.** A 3-file markdown diff cannot break a Worker build. The same class already appears earlier today in this log's Wave-6 entry lineage — `156e606e` logged `Data Worker: Deployment cancelled` at 18:18:59Z. Six container cycles across two trains inside ~20 minutes is exactly when Railway drops one.
  - ⚠️ **Observed cache gap — RTH restarts:** 2 pushes in the 18:36–18:40Z window each cycle the Worker container. Because the `3397e070` Worker build **failed**, the running container was **not** replaced by it — the only Worker restart that completed was the `02ac05b2` one finishing 18:37:33Z. Treat roughly **18:36–18:38Z** as a deploy artifact window and exclude it when reading today's intraday paired-% / alert-lag; do not chase it as a data bug. Live-trade **DORMANT** → no live money at risk.
  - 🔴 **OPEN AT HANDOFF — owed to E/M, deliberately NOT done by R:** deciding whether to re-trigger the `Worker` deploy so it lands `3397e070`. R has **no `railway` authority** (hard line: never `railway variables` / `railway up`) and did not revert, reset, or force-push. R's read — since the delta is two docs files and the service is healthy on `02ac05b2`, this is **safe to leave until post-close** rather than cycling the Worker again mid-RTH (each RTH restart costs ~5 edges/active strategy). E's call, not R's.
  - Gates (R·auto): none run, and correctly so — markdown-only train, briefed as such. Pre-merge validation instead: both branches confirmed `MERGEABLE`/`CLEAN`; `git merge-base` for **both** = `cb1a228e` exactly, i.e. cut from the then-current dev head with nothing dragged in (board #153 contract); GitHub `synthetic-parity` + `build` verified **green on each PR's final head via the `/check-runs` API**, not merely the rollup; #126 re-confirmed `MERGEABLE`/`CLEAN` after #125 advanced dev to `02ac05b2`; zero file overlap verified. Backup: **`backup/dev-pre-wave7-0729`** @ `cb1a228e`, pushed and verified on the remote before the first merge.
  - 📋 **Brief-vs-reality discrepancy (flagged, not fatal):** the release brief predicted #126 would carry **2** commits; it carried **4** (`ca9f633e`, `6db31715`, `b026c84b`, `8df15ac1`). R inspected each individually — all four are M's own docs commits and all four touch **only** the two files the brief named. Net diff matched the brief exactly, so the check's actual purpose (no unrelated commits riding along) passed and R proceeded. The brief had simply undercounted, naming only the post-open commit.
  - **No RORT_* flag and no branch-protection / merge-queue setting was armed** — `RORT_T0_HEALTH_SNAPSHOT`, `RORT_HEALTH_HB_SUPERSEDES_SNAPSHOT` (#157), #121 `MTF_PB_PREV_EPOCH` and #125 settle-retrue all left exactly as they were; #166 branch-protection / merge-queue arming (steps 3–4) still deferred per Kevin's standing instruction.

- **18:16:57 + 18:17:13 + 18:17:24 + 18:17:36 UTC (12:16/12:17 MT)** — `80770532` + `e8c96a5b` + `156e606e` + `cb1a228e` **Wave-6 train — four cars, all shipped** (board #190, R·auto). ⚠️ **Market OPEN** — Wednesday 07-29, merges landed mid-RTH (13:30–20:00Z), **not** post-close like Waves 1–5; live-trade **DORMANT**. Merge order = brief order, docs first / code last so a failure on the code car would leave the docs already landed: PR #121 `docs/deploy-log-wave5-0729` (the owed Wave-5 Deploy_Log entry; board #187; `Deploy_Log.md` only, +10/−0), PR #122 `docs/m-handoff-0729` (M roster handoff — retires M `c85c7f28`, adds successor `9f1e8099`; `Session_Charters.md` only, +2/−1; **this is the car that clears preflight check (4) M-lane charter drift**), PR #124 `docs/tm-gate-verification-182` (TM gate verification review — 2 blockers, design retired in favour of the inbound check; new `docs/_active/TM_Gate_Verification_2026-07-29.md`, +146/−0), then PR #123 `feat/inbound-check-182` (board #182 **Step 7** — team-dispatcher **V4.17 → V4.18**, the **inbound check**: each dispatched agent is first asked to verify the PREVIOUS chain step actually delivered what its SOP promised, and to STOP + reassign to M rather than build on a bad hand-off; `tools/team_dispatcher/dispatcher.py` plus a new standalone test `src/test_dispatcher_inbound_check_182.py`). **Zero file overlap** across all four; final dev head `cb1a228e` (parents `156e606e` + PR-123 head `feefaed4` — the exact head validated pre-merge).
  - Service(s) redeployed: api + Worker + frontend + batch-worker + Data Worker + Streamlit + flat-file-cron (dev auto-deploy, watchPatterns=[] → all rebuild on any dev push), 4 pushes coalescing/superseding to the final `cb1a228e` build. **shadow-worker NOT redeployed** (off this path). **No RORT_* flag touched** (a git push cannot alter env-var flag config).
  - **Runtime impact on the deployed services is ZERO:** #121/#122/#124 are docs; #123 edits ONLY `tools/team_dispatcher/dispatcher.py` — the **local team-dispatcher loop that runs on Kevin's machine, NOT a deployed Railway service** — and its test. No api router, no frontend, no engine/data-loading/resample/warmup/cache path is touched — hence the heavy `fidelity_parity_suite.py` was correctly NOT required (path-based gating per its own docstring scope; the hermetic Fidelity Gate / `synthetic-parity` GitHub Action is the gate that applies, and it was green). Every PR's file list was pulled from GitHub and checked against that scope rather than assumed.
  - 🔴 **Dispatcher restart OWED to M (NOT performed by R):** #123 modifies the very dispatcher loop that was running the headless R process, so restarting it mid-run would have orphaned R. The running loop holds the OLD code in memory — **V4.18's inbound check does not take effect until that loop is restarted**, and until then preflight check (5) will correctly report the running dispatcher's file differs from dev. M pulls the main checkout to `cb1a228e` and restarts: `nohup setsid python3 tools/team_dispatcher/dispatcher.py --live --loop --poll 20 > /tmp/dispatcher.log 2>&1 &` (same handling as the #120 / V4.17 train earlier today).
  - **No DB migration in this train** (three docs cars + one local-tooling car; no schema change).
  - ⚠️ **Observed cache gap — RTH restarts, unlike every prior wave:** 4 pushes in the 18:16–18:18Z window each cycle the Worker container (~1–2 min warmup per restart, coalesced by Railway to the final `cb1a228e` build; Worker reported SUCCESS 18:19:09Z). **The market was OPEN**, so bar-recording / live-alert gaps in roughly the **18:17–18:20Z** window are deploy artifacts and must not be chased as data bugs. Live-trade is **DORMANT** → no live money at risk, but this window should be excluded when reading today's intraday paired-% / alert-lag.
  - Post-deploy verification (R·auto, GitHub Railway commit-statuses): **all 7 services report SUCCESS on `cb1a228e`** (Data Worker 18:18:57Z, flat-file-cron 18:18:58Z, Streamlit 18:18:59Z, batch-worker 18:19:06Z, api 18:19:08Z, Worker 18:19:09Z, frontend 18:19:31Z; combined status state = `success`). CI on the merge commit: `synthetic-parity` **success**, `Supabase Preview` skipped. No deployed runtime code changes in this train (dispatcher.py is a local tool; the other three cars are docs), so there is **no edge fingerprint to observe** — Railway SUCCESS across all services is the appropriate confirmation for a docs / local-tooling train, and the brief scoped deploy-watch to "confirm it is not red".
  - Gates (R·auto, all PASS — **path-based, not blanket**): run from the CLEAN `/home/kevin/projects/Kevbot-release` worktree (never the main checkout), detached at PR-123 head `feefaed4`. First verified `git log feefaed4..origin/dev` was **EMPTY** — i.e. `feefaed4` is a strict superset of dev, so testing that SHA tested the **exact** merge result — and `origin/dev..feefaed4` showed only its own 2 commits (clean base, board #153). `src/test_dispatcher_inbound_check_182.py` **18 passed / 0 failed**; `src/test_dispatcher_step_level_182.py` **ALL PASS (21 checks)**; `tools/team_dispatcher/test_dispatcher_reap.py` **ALL PASS**; `src/test_dispatcher_loud_skips_171.py` **ALL PASS (15 checks)**. Confirmed the two board-writing suites stub `api` (hermetic — no writes to the real board). **All four re-run against the merged dev tip `cb1a228e`** afterwards and still green (18/0 · 21 · ALL · 15). Backup: **`backup/dev-pre-wave6-0729`** @ `bbaf67f2`, pushed and verified on the remote before the first merge.
  - **No RORT_* flag and no branch-protection / merge-queue setting was armed** — `RORT_T0_HEALTH_SNAPSHOT`, `RORT_HEALTH_HB_SUPERSEDES_SNAPSHOT` (#157), #121 `MTF_PB_PREV_EPOCH` and #125 settle-retrue all left exactly as they were; #166 branch-protection / merge-queue arming (steps 3–4) still deferred per Kevin's standing instruction.

- **03:55:08 + 03:55:52 UTC (21:55 MT 07-28)** — `b3606fc7` + `bbaf67f2` **Wave-5 train — two cars** (board #187, R·auto). Market **CLOSED** (post-close; RTH ended 20:00Z 07-28), live-trade **DORMANT**. Merge order = brief order: PR #119 `docs/deploy-log-wave4-0729` (the owed Wave-4 Deploy_Log entry; board #186; docs only — verified it accurately records the two-car Wave-4 train `832e2bf2` + `004bf85b` ending at `004bf85b`, so no amend was needed), then PR #120 `feat/dispatcher-step-level-182` (board #182 **Step 7** — team-dispatcher **V4.16 → V4.17**, step-level dispatch: sends the agent the **current process-chain step + that step's own SOP**, gates on the current step's owner, refuses discuss-mode / scope-drift steps; `tools/team_dispatcher/dispatcher.py` +99/−4 plus a new standalone test `src/test_dispatcher_step_level_182.py`). **Zero file overlap** (#119 = `Deploy_Log.md` only; #120 = `dispatcher.py` + its test only); final dev head `bbaf67f2` (parents `b3606fc7` + PR-120 head `f1ceea92` — the exact head validated pre-merge).
  - Service(s) redeployed: api + Worker + frontend + batch-worker + Data Worker + Streamlit + flat-file-cron (dev auto-deploy, watchPatterns=[] → all rebuild on any dev push), superseding to the `bbaf67f2` build. **shadow-worker NOT redeployed** (off this path). **No RORT_* flag touched** (a git push cannot alter env-var flag config).
  - **Runtime impact on the deployed services is ZERO:** #119 is docs; #120 edits ONLY `tools/team_dispatcher/dispatcher.py` — the **local team-dispatcher loop that runs on Kevin's machine, NOT a deployed Railway service** — and its test. No api router, no frontend, no engine/data-loading/resample/warmup/cache path is touched — hence the heavy `fidelity_parity_suite.py` was correctly NOT required (path-based gating; the hermetic Fidelity Gate / `synthetic-parity` GitHub Action is the gate that applies, and it was green on both branches).
  - 🔴 **Dispatcher restart OWED to M (NOT performed by R):** #120 modifies the very dispatcher loop that was running the headless R process, so restarting it mid-run would have orphaned R. M pulls the main checkout to `bbaf67f2` and restarts the loop after the train completes (same handling as the #171 train on 07-28). Until M restarts, the live dispatcher keeps running V4.16 behaviour; V4.17 step-level dispatch goes live only on that restart.
  - **No DB migration in this train** (both cars are docs / local-tooling; no schema change).
  - Observed cache gap: 2 pushes in the 03:55–03:56Z window each cycle the Worker container (~1–2 min warmup per restart, coalesced/superseded by Railway to the final `bbaf67f2` build). **Market CLOSED + live-trade DORMANT** → no live money at risk, no live alerts lost.
  - Post-deploy verification (R·auto, GitHub Railway commit-statuses + HTTP): **all 7 services report SUCCESS on `bbaf67f2`** (Data Worker, flat-file-cron, batch-worker, Worker, Streamlit, api, frontend; commit-status `success` 03:56:11–03:56:22Z, combined state `success`). api `/health` **200**; `/api/dev-tasks` **401** unauth (router loaded + admin-gated, **not 5xx** → deployed without an import/startup crash); frontend `/admin/tasks` **200**. Neither car changes deployed runtime code (dispatcher.py is a local tool; Deploy_Log.md is docs), so there is **no edge fingerprint to observe** — the Railway SUCCESS + continued 200s is the appropriate confirmation for a docs / local-tooling train.
  - Gates (R·auto, all PASS — **path-based, not blanket**): re-run from a CLEAN `origin/dev` worktree (merge-preview of dev@`004bf85b` + each PR head): PR #120 standalone `src/test_dispatcher_step_level_182.py` **ALL PASS (21 checks)**, regression `src/test_dispatcher_loud_skips_171.py` **15/15**, `src/test_stamp_ticks_kevin_step_151.py` **14/14** (no dispatcher regression); each PR re-confirmed MERGEABLE/CLEAN against the advancing dev head (#120 re-checked after #119 advanced dev to `b3606fc7`); **zero file overlap** verified; GitHub `synthetic-parity` + `build` green on both PRs. Backup: **`backup/dev-pre-wave5-2026-07-28`** @ `004bf85b` (created server-side; R is headless and cannot local-push).
  - **No RORT_* flag and no branch-protection / merge-queue setting was armed** — `RORT_T0_HEALTH_SNAPSHOT`, `RORT_HEALTH_HB_SUPERSEDES_SNAPSHOT` (#157), #121 `MTF_PB_PREV_EPOCH` and #125 settle-retrue all left exactly as they were; #166 branch-protection / merge-queue arming (steps 3–4) still deferred per Kevin's standing instruction (deferred until #182 lands).

- **00:44:57 + 00:45:38 UTC (18:44/18:45 MT 07-28)** — `832e2bf2` + `004bf85b` **Wave-4 train — two cars** (board #186, R·auto). Market **CLOSED** (post-close; RTH ended 20:00Z 07-28), live-trade **DORMANT**. Merge order = brief order: PR #117 `docs/deploy-log-wave3-0728` (the owed Wave-3 Deploy_Log entry; board #185; docs only — verified it already records the full **five-car** Wave-3 train ending at `d38cce4a` incl the Car-5 resolution note, so no amend was needed), then PR #118 `feat/process-chain-accordion-182` (board #182 Steps 4–6 — the front-end half of the process-chain feature: summary-tab accordion step rendering + step actions + slash-command insert; two `.tsx` files `TaskDetailModal.tsx` + `taskBoardShared.tsx`, +730/−21; **NO api, NO migration, NO engine path**). **Zero file overlap** (#117 = `Deploy_Log.md` only; #118 = the two board `.tsx` only); final dev head `004bf85b` (parents `832e2bf2` + PR-118 head `4fde01f5` — the exact head validated pre-merge).
  - Service(s) redeployed: api + Worker + frontend + batch-worker + Data Worker + streamlit + flat-file-cron (dev auto-deploy, watchPatterns=[] → all), superseding to the `004bf85b` build. **shadow-worker NOT redeployed** (off this path). **No RORT_* flag touched** (a git push cannot alter env-var flag config).
  - **Runtime impact is frontend-only:** #117 is docs; #118 touches only two board-view `.tsx` files (board UI). No api/engine/data-loading/resample/warmup/cache path — hence the heavy `fidelity_parity_suite.py` was correctly NOT required (path-based gating; the hermetic Fidelity Gate / `synthetic-parity` GitHub Action is the gate that applies, and it was green on both branches).
  - **No DB migration in this train** (#118 is pure frontend; #114's `dev_task_comments.step_order` column was already applied 07-28 23:41Z with the Wave-3 train).
  - Observed cache gap: 2 pushes in the 00:44–00:46Z window each cycle the Worker container (~1–2 min warmup per restart, coalesced/superseded by Railway to the final `004bf85b` build). **Market CLOSED + live-trade DORMANT** → no live money at risk, no live alerts lost.
  - Post-deploy verification (R·auto, GitHub Railway commit-statuses + HTTP): **all 7 services report SUCCESS on `004bf85b`** (api, Worker, Data Worker, batch-worker, Streamlit, flat-file-cron, frontend). api `/health` **200**; `/api/dev-tasks` **401** unauth (router loaded + admin-gated, **not 5xx** → the merge deployed without an import/startup crash); frontend `/admin/tasks` **200**, `/admin/strategy-health` **200**, `/` **307** (login redirect). **Caveat — the #118 board bundle was NOT yet observable at the edge within the lease window:** the unauthenticated `/admin/tasks` HTML is served from Next's full-route cache (`cache-control: s-maxage=31536000, x-nextjs-cache: HIT`) and still references the pre-#118 chunk manifest (board chunk `4347-73f3023a22d10279.js` unchanged; the `'audible'` string literal absent from every served chunk; a query-param cache-bust still returned HIT). The frontend **container** deployed SUCCESS and authenticated admin loads bypass the static route cache, so this is a CDN-cache/edge-latency artifact, **not a deploy failure** — consistent with prior trains. The browser-driven confirmation (log in → `/admin/tasks` → open a task-detail modal → open board **#182** → confirm its chain renders as **accordion steps** with **two 🔀 audible** steps, the feature's own first test case) needs an authenticated admin session (unavailable this headless run; documented dev account is non-admin) — **handed to Kevin/M to eyeball.**
  - Gates (R·auto, all PASS — **path-based, not blanket**): (1) each branch re-checked MERGEABLE/CLEAN against the advancing dev head before its merge (#118 re-confirmed CLEAN after #117 advanced dev to `832e2bf2`); (2) **zero file overlap** verified across both PRs; (3) #118 standalone `src/test_process_chain_182.py` **ALL PASS (37 checks)** run against a clean origin/dev merge-preview (`d38cce4a` + PR head `4fde01f5`, clean no-conflict merge); (4) #118 `tsc --noEmit` merge-preview vs dev-head control in one isolated worktree (symlinked node_modules, same env both runs) = **40 == 40**, sorted error-sets byte-identical, **zero new errors**, none in the two #118 files (the 40 are pre-existing missing-module env errors; the brief cited 45 from a different node_modules/@types env — the **delta**, which is what gates a frontend car, is zero); (5) GitHub `synthetic-parity` + `build` **pass** on both PRs. Backup: **`backup/dev-pre-wave4-2026-07-28`** @ `d38cce4a` (created server-side; R is headless and cannot local-push).
  - **No RORT_* flag and no branch-protection / merge-queue setting was armed** — `RORT_T0_HEALTH_SNAPSHOT`, `RORT_HEALTH_HB_SUPERSEDES_SNAPSHOT` (#157), #121 `MTF_PB_PREV_EPOCH` and #125 settle-retrue all left exactly as they were; #166 branch-protection / merge-queue arming (steps 3–4) still deferred (Kevin's announce-first rule).

## 2026-07-28


- **22:04:16 + 22:05:04 + 22:05:41 + 22:06:19 UTC (16:04/16:05/16:05/16:06 MT)** — `b24fe249` + `b604c42f` + `75b65b03` + `e63deb62` **Wave-3 train — all five cars shipped** (board #185, R·auto; the first four cars pushed 22:04–22:06Z, Car 5 held at 22:07Z pending its DB migration, then resolved and merged 00:12Z 2026-07-29 — see the ✅ resolution note at the bottom of this entry). Merge order = brief order: PR #112 `docs/deploy-log-wave2-0728` (the owed Wave-2 Deploy_Log entry; board #181; docs only), PR #113 `docs/pm-tm-roles` (board #182 — charter retitles M → **Project Manager** and adds a **TM / Task Manager** role to the registry; TM is planned-only, NOT built = step 6 of #182; docs only), PR #115 `feat/ci-merge-queue-gates-166` (board #166 steps 1–2 — CI-workflow files only: diff-scopes the engine suite in `fidelity-gate.yml` and adds a new `frontend-gate.yml`; both wired to also run on `merge_group` so they *can* later serve as REQUIRED checks — but **nothing is marked REQUIRED and branch protection / the merge queue were NOT armed**, those are #166 steps 3–4, settings flips, deliberately deferred), PR #116 `fix/health-snapshot-stale-dead-alarm-157` (board #157 — health router `strategy_health.py` consults fresh `shadow_heartbeats` instead of the deprecated `engine_snapshot_at` for the "0 healthy" dead-alarm, all behind default-OFF `RORT_HEALTH_HB_SUPERSEDES_SNAPSHOT`). **Zero file overlap** across all four; final dev head `e63deb62`.
  - **PR #114 `feat/process-chain-schema-182` (board #182 Step 3) was HELD at 22:07Z, then merged at 00:12Z 2026-07-29 as `d38cce4a`** once its migration was applied — see the ✅ resolution note at the bottom of this entry.
  - Service(s) redeployed: api + Worker + frontend + batch-worker + Data Worker + streamlit + flat-file-cron (dev auto-deploy, watchPatterns=[] → all), once per push, superseding to the final `e63deb62` build. **shadow-worker NOT redeployed** (off this path — no branch/commit on its latest deployment, consistent with prior trains). **No RORT_* flag touched** (a git push cannot alter env-var flag config); **#116's new `RORT_HEALTH_HB_SUPERSEDES_SNAPSHOT` left default-OFF**, and **#121 `MTF_PB_PREV_EPOCH` + #125 settle-retrue left exactly as armed** — both mid-validation, out of scope. **No branch-protection / merge-queue setting was armed** (#166 arming is deferred).
  - **Runtime impact is a single flag-gated health-router change despite the all-services rebuild:** #112 + #113 are docs; #115 is CI-workflow YAML only (`.github/workflows/**` — never a deployed request path); #116 is a +63-line edit to the read-only `strategy_health.py` reporting router whose new behavior is fully behind the OFF `RORT_HEALTH_HB_SUPERSEDES_SNAPSHOT` switch (no decision path, no engine/data/resample/warmup/cache code touched). Hence the heavy `fidelity_parity_suite.py` was correctly NOT required (path-based gating; the hermetic Fidelity Gate / `synthetic-parity` GitHub Action is the gate that applies, and it was green on all four branches).
  - **No DB migration was applied during the initial four-car push (22:04–22:07Z)** — the one migration in scope was applied later, at 23:41Z (see the resolution note). At push time, the migration — #114's `src/migrations/dev_tasks_process_chain.sql` (`ALTER TABLE dev_task_comments ADD COLUMN IF NOT EXISTS step_order INTEGER`, additive + idempotent) — was **NOT run**: applying prod DDL was permission-denied for a headless dispatch (needs Kevin's own authorization), and #114 was held rather than shipped against a DB missing the column (see bottom note). A read-only probe confirmed `dev_task_comments.step_order` did not yet exist at push time (applied 23:41Z — see the resolution note).
  - Observed cache gap: 4 pushes in the 22:04–22:07Z window each cycle the Worker container (~1–2 min warmup per restart, coalesced/superseded by Railway to the final `e63deb62` build). **Market was CLOSED** (post-close; RTH ended 20:00Z) and **live-trade is DORMANT** — no live money at risk, no live alerts lost.
  - Post-deploy verification (R·auto, HTTP + read-only `railway status --json`): **all 8 services report `status=SUCCESS` on commit `e63deb62`** (api, Worker, Data Worker, batch-worker, streamlit, flat-file-cron, frontend; shadow-worker SUCCESS/off-path). api `/health` **200**; `/api/admin/strategy-health` + `/api/admin/strategy-health/backlog` + `/api/dev-tasks` all **401** unauthenticated = routers loaded + admin-gated + alive (**not 5xx** → the #116 router change did not crash import/startup). Frontend `/admin/tasks` **200**, `/admin/strategy-health` **200**, `/` **307** (login redirect). The browser-driven "task-detail modal opens" confirmation was NOT performed — it needs an authenticated admin session (Kevin's JWT), unavailable this headless run — **handed to Kevin/M to eyeball once convenient** (low risk: both pages serve 200, the API is 401-not-5xx, the frontend container deployed SUCCESS, and #116 passed its hermetic test).
  - Gates (R·auto, all PASS — **path-based, not blanket**): (1) each branch re-checked MERGEABLE/CLEAN against the advancing dev head before its merge; (2) **zero file overlap** verified across all five PRs; (3) hermetic tests re-run from a clean origin/dev worktree — #116 `test_health_snapshot_stale_157.py` **12 passed** (pytest), #114 `test_process_chain_182.py` **ALL PASS (37 checks)** (standalone); (4) GitHub `synthetic-parity` **pass** on every PR, #115 `build` **pass**; (5) #115 YAML inspected — adds workflow definitions only, arms no REQUIRED check. Backup: **`backup/dev-pre-wave3-2026-07-28`** @ `84d1d2a5` (created server-side; R is headless and cannot local-push).
  - 🛑 **PR #114 `feat/process-chain-schema-182` HELD (board #182 Step 3, board #185 Car 5).** #114 adds the process-chain step schema + server-side transition rules to the `dev_tasks` API, and its `add_comment` **auto-stamps `dev_task_comments.step_order` on every insert**. Its migration adds that column. Applying the prod DDL was **permission-denied for the headless run** (a production DB migration needs Kevin's own [named+specifics] authorization, not a relayed dispatch brief); a read-only probe confirmed the column is **absent**. Merging #114 without the column would deploy an API that 500s **every comment post on the live board**, so per "never ship past a red gate" the merge was held — cars 1–4 are independent of #114 and shipped clean, so no rollback was needed. **#114 is ready to merge the instant the column exists**; its 37-check hermetic test already passes. Next action: Kevin applies (or authorizes) the one-line additive/idempotent migration, then #114 is merged via a re-dispatched R. Board #185 reassigned to **kevin** with this ask.
  - ✅ **Car 5 RESOLVED + SHIPPED — 2026-07-29 00:12Z (board #185, re-dispatched R·auto).** Kevin authorized the migration and M applied it at 2026-07-28 23:41Z: `ALTER TABLE dev_task_comments ADD COLUMN IF NOT EXISTS step_order INTEGER` (0.08s; additive, nullable, no default → metadata-only, no table rewrite, 521 rows unaffected; verified `('step_order','integer','YES')`). A fresh R·auto re-validated #114 from a clean `origin/dev` worktree — standalone `test_process_chain_182.py` **ALL PASS (37 checks)** on the real merge preview (`e63deb62` + PR head `46f88315`, clean no-conflict merge), CI `synthetic-parity` green, `mergeable=CLEAN` — and merged it as **`d38cce4a`** (`Merge pull request #114 from ktkventures/feat/process-chain-schema-182`, 00:12:03Z; parents `e63deb62` + `46f88315`). **Final Wave-3 dev head = `d38cce4a`.** Deploy verified ~00:14Z: the api service serves #114's new `POST /api/dev-tasks/{id}/steps/{complete,raise-issue,stamp}` routes (401 unauth = mounted; bogus sibling paths return 404/405 = per-route auth, so the 401 proves the NEW code booted, not blanket middleware); api `/health` **200**, `/api/dev-tasks/{id}/poll` + `/comments` **401-not-5xx** (closes the migration-window risk — a missing `step_order` column would 500 these), frontend `/admin/tasks` + `/admin/strategy-health` **200**; `dev_task_comments.step_order` confirmed present + selectable via PostgREST. **No RORT_* flag and no branch-protection / merge-queue setting was armed** (#116's `RORT_HEALTH_HB_SUPERSEDES_SNAPSHOT`, #121 `MTF_PB_PREV_EPOCH`, #125 settle-retrue, and `RORT_T0_HEALTH_SNAPSHOT` all left exactly as they were; #166 arming still deferred). The browser-driven "task-detail modal opens" eyeball still needs an authenticated admin session (unavailable headless) — handed to Kevin/M.

- **19:09:42 + 19:10:5x + 19:11:5x + 19:18:59 UTC (13:09/13:10/13:11/13:18 MT)** — `4034e52b` + `85d200c0` + `cc123bef` + `84d1d2a5` **Wave-2 train, four cars** (board #181, R·auto). Merge order = brief order: PR #108 `docs/deploy-log-wave1-0728` (the owed Wave-1 Deploy_Log entry; board #179; docs only), PR #109 `fix/dangling-alert-annotate-117` (board #117 Layer 1 — annotate-only closure for dangling alert-history "Open" rows: offline sweeper `annotate_orphaned_alerts.py` + the additive `alerts_orphaned_columns.sql`), PR #110 `feat/t0-health-snapshot-160` (board #160 — daily T+0 health snapshot: `t0_health_snapshot.py` + a +6-line `data_worker.py` thread-spawn + 16 tests, all behind default-OFF `RORT_T0_HEALTH_SNAPSHOT`), PR #111 `feat/last10-unified-sim-177` (board #177, folds #175 — unified last-10 modal with fired/theo/SIM tabs: retires `SimBasisModal` into an exported `SimPanel`, adds read-only `GET /api/strategy-health-sim/{sid}?detail=true`; 5 files, +825/-432). **Zero file overlap** across all four; final dev head `84d1d2a5`.
  - Service(s) redeployed: api + Worker + frontend + batch-worker + Data Worker + streamlit + flat-file-cron (dev auto-deploy, watchPatterns=[] → all), once per push, superseding to the `84d1d2a5` build. **shadow-worker NOT redeployed** (off this path — last deploy 07-23). **No RORT_* flag touched** (a git push cannot alter env-var flag config); **`RORT_T0_HEALTH_SNAPSHOT` left default-OFF** (its capture thread is inert until armed) and **#121 `MTF_PB_PREV_EPOCH` + #125 settle-retrue left exactly as armed** — both mid-validation, out of scope.
  - **Runtime impact is frontend + read-only-API only despite the all-services rebuild:** #108 is docs; #109's `annotate_orphaned_alerts.py` is an offline, dry-run-first operator sweeper (never a deployed request path) and its migration was already live; #110's `data_worker.py` change is +6 lines that only `start_capture_thread(...)` behind the OFF flag (no decision path reached); #111 is the health-page frontend + two read-only health/sim routers (`health_last10.py` +10 lines, `replay_sim.py` +158). **No engine, data-loading, resample, warmup or cache code path was touched** — hence the heavy `fidelity_parity_suite.py` was correctly NOT required (path-based gating; the hermetic Fidelity Gate / `synthetic-parity` GitHub Action is the gate that applies, and it was green on all four branches).
  - **DB migration applied to prod (~19:15Z, R·auto, direct psycopg3 / service-role):** `src/migrations/t0_health_snapshots.sql` (#110) — NEW `t0_health_snapshots` table (additive, RLS **ON** with no public policy → service-role only), `UNIQUE(snapshot_date, strategy_id)`, indexes `idx_t0_health_sid_date` + `idx_t0_health_date` + pkey, 34 columns, 0 rows at apply. Idempotent (`CREATE TABLE/INDEX IF NOT EXISTS`); verified `to_regclass` None→present and `relrowsecurity=true` post-apply. **#109's `alerts_orphaned_columns.sql` was NOT re-run by R** — M had already applied it ~18:15Z (both columns added + index built with `CREATE INDEX CONCURRENTLY` to avoid a write-blocking lock on the 512,450-row `alerts` table during RTH; sweeper already ran `--apply`, 2,555 rows stamped, 0 `exit_signal` synthesized). The committed file still carries a plain `CREATE INDEX` — follow-up #180 tracks fixing the file itself.
  - Observed cache gap: 4 pushes in the 19:09–19:19Z window each cycle the Worker container (~1–2 min warmup per restart, coalesced/superseded by Railway to the final `84d1d2a5` build). **Market was OPEN (RTH; closes 20:00Z) and Kevin explicitly approved deploying during RTH today** — none of these cars touches a decision path, and **live-trade is DORMANT** (no live money at risk); brief intermittent warmup gaps possible in that window.
  - Post-deploy verification: Railway GitHub commit-status for head `84d1d2a5` = **SUCCESS across all seven services** (all `pending` 19:18:5xZ → all `success` 19:19:31–19:20:39Z: flat-file-cron 19:19:31, Data Worker 19:19:33, api 19:19:37, Worker 19:19:43, batch-worker 19:19:53, Streamlit 19:19:57, frontend 19:20:39). api `/health` **200** held continuously; frontend `/admin/strategy-health` **200** serving the genuine RoR Trader Next.js shell (`x-nextjs-cache: HIT`, railway edge). `openapi.json` is disabled in prod (404), so no schema-diff probe was possible. **The browser-driven "last-10 modal opens with its fired/theo/SIM tabs" confirmation was NOT performed** — the Playwright MCP was not exposed this headless run and the documented dev test account is non-admin (no `/admin` access) — **handed to F/Kevin to eyeball once convenient** (low risk: the health page serves 200, the frontend container deployed SUCCESS, and #111 compiles clean).
  - Gates (R·auto, all PASS — **path-based, not blanket**): (1) each branch re-checked MERGEABLE/CLEAN against the advancing dev head before its merge; (2) **zero file overlap** verified across all four PRs; (3) #110 16 unit tests green in CI (`synthetic-parity` PASS); (4) #111 `tsc --noEmit` PR-head vs dev-head control in one isolated worktree (symlinked node_modules, same env both runs; no earlier car touched frontend) = **45 == 45**, sorted error-sets byte-identical, **zero new errors**, none in the #111 files (the 45 are pre-existing missing-module env errors — drifted up from the 40 on 07-27 via unrelated merges, not this train); (5) GitHub `synthetic-parity` **pass** on every PR; (6) migration verified post-apply. Backup: **`backup/pre-wave2-0728`** @ `9af67cc0` (created server-side; R is headless and cannot local-push).

- **17:28:53 + 17:31:15 + 17:35:39 + 17:36:49 UTC (11:28/11:31/11:35/11:36 MT)** — `ff4964d3` + `be7e032d` + `d68d79a8` + `9af67cc0` **Wave-1 train, four cars** (board #179, R·auto). Merge order = brief order: PR #104 `docs/deploy-log-0727-flane` (the two owed 07-27 Deploy_Log entries — R·auto's #99/#100 train + the F-lane train; docs only), PR #106 `fix/local-update-env-parity-161` (board #161 — `local_update.py` now asserts flag + dependency parity against the batch-worker RORT_ manifest and fails CLOSED before any prod write), PR #107 `feat/handoff-chain-avatars-169` (board #169 — board card surfaces the handoff chain as an owner-avatar sequence; frontend only, 3 `.tsx`), PR #105 `feat/dispatcher-assignee-authoritative-171` (board #171, folds #170 — assignee is authoritative, the `next_actor` checklist gate is deleted, dispatch skips are LOUD + deduped). Each car = 1 commit / 3 files (`#104` = 1 file), **zero file overlap**, final dev head `9af67cc0`.
  - Service(s) redeployed: api + Worker + frontend + batch-worker + Data Worker + streamlit + flat-file-cron (dev auto-deploy, watchPatterns=[] → all), once per push, superseding to the `9af67cc0` build. **shadow-worker NOT redeployed** (off this path — last deploy 07-23). **No RORT_* flag touched** (a git push cannot alter env-var flag config); **#121 `MTF_PB_PREV_EPOCH` and #125 settle-retrue left exactly as armed** — both are mid-validation and were not in scope.
  - **Runtime impact is frontend-only despite the all-services rebuild:** #104 is docs, #106 edits `src/local_update.py` (a LOCAL operator tool that never ships to Railway), #105 edits `tools/team_dispatcher/dispatcher.py` (runs on Kevin's machine, NOT Railway). The only functional change to a *deployed* service is #107's board UI. No engine, worker, data-loading, resample, warmup or cache code path was touched — hence the heavy `fidelity_parity_suite.py` was correctly NOT required (path-based gating; the hermetic `synthetic-parity` GitHub Action is the gate that applies, and it was green on all four branches).
  - Observed cache gap: 4 pushes in the 17:29–17:37Z window each cycle the Worker container (~1–2 min warmup per restart, likely coalesced/superseded by Railway to the final `9af67cc0` build). Market was OPEN (RTH, closes 20:00Z) but **live-trade is DORMANT** — no live money at risk; brief intermittent warmup gaps possible during that window. R·auto's HTTP deploy-watch began ~17:40Z (after the last merge) and so did not instrument the individual restart dips.
  - Post-deploy verification (HTTP-only, no railway CLI — R·auto is permission-blocked from it): API `/health` **200** and `/api/docs` **200** held continuously across the watch; board API `/api/dev-tasks` **401** unauthenticated = admin-gated and alive; frontend `/admin/tasks` **200**. Frontend deploy fingerprint: across a 6-min HTTP watch (~17:40–17:47Z, ~4–11 min post-merge) **none of the 13 statically-referenced chunk hashes changed**, and the deployed board bundle (`chunks/4347-73f3023a22d10279.js`, 196 KB) still carries #143's `Messages` code from the 07-27 train but shows **no #107 `handoff` markers** — i.e. #107's board UI had **not yet gone live** inside the lease window. Read as Railway frontend-build latency (four rapid pushes coalescing to `9af67cc0`), **not a failure**: the merge is on `dev` and the api side deployed and serves, so the frontend container will ship #107 once its build finishes. The browser-driven "task detail modal opens" confirmation needs Playwright (MCP unavailable this run) + admin auth, so it could not be performed headless — **handed to M to eyeball the avatar chain + a modal after the frontend container settles (and after M's main-checkout pull + dispatcher restart).**
  - Gates (R·auto, all PASS — **path-based, not blanket**): (1) commit-range per branch vs `origin/dev` = **exactly 1** for all four; (2) #106 pytest `test_local_update_env_parity.py` = **14/14** (with the real `src/.env` present — a fresh worktree lacks the gitignored `.env`, which is the only reason a bare run shows 13/1; the failing case is the real-env mirror check and it PASSES against Kevin's actual `.env`, which also proves the local mirror is currently in lock-step with the manifest); (3) #107 `tsc --noEmit` PR-head vs dev-head control in one worktree = **40 == 40**, sorted error-sets byte-identical, **zero new errors**, none in the 3 train files; (4) #105 standalone scripts `test_dispatcher_loud_skips_171.py` **15/15** + `test_stamp_ticks_kevin_step_151.py` **14/14**; (5) GitHub `synthetic-parity` **pass** on every PR. Backup: **`backup/dev-pre-wave1-0728`** @ `1475be66` (pushed).
  - 🔴 **Dispatcher restart OWED to M:** #105 changed `dispatcher.py` while the live dispatch loop was running R·auto itself; R·auto deliberately did **not** restart it (self-orphan). M pulls the main checkout to `9af67cc0` and restarts the loop to pick up the assignee-authoritative logic — this is why #105 merged last. Preflight AFTER flags this as invariant (5) "dispatcher file DIFFERS from dev" and (1) "main checkout 8 behind origin/dev"; both clear on M's pull + restart.

## 2026-07-27

- **22:06:5x + 22:12:3x + 22:18:3x UTC (16:06/16:12/16:18 MT)** — `fa282303` + `310a8a8d` + `04b4c47d` **F-lane train, three cars** (board #174, R — release f-lane 07-27): PR #101 `fix/board-stamp-and-poll-151-148` (board #151 stamp ticks the pending Kevin checklist step — the bug that left the ENTIRE Todo queue undispatchable ~41h — plus board #148, the modal's 7s/3–4-call poll consolidated into ONE 15s `/poll` call that board-refetches only on real change), PR #102 `feat/mentions-messages-tab-143` (board #143 @-mentions fan-out at comment-POST time + unread Messages inbox tab), PR #103 `feat/sim-basis-ui-144` (board #144 V2.13 SIM basis UI — on-demand Run, SIM column/badge, per-strategy nightly opt-in; frontend only).
  - Service(s) redeployed: api + Worker + frontend + batch-worker + Data Worker + streamlit + flat-file-cron (dev auto-deploy, watchPatterns=[] → all), once per push; the `fa282303` set was superseded ~5 min later by `310a8a8d`. **shadow-worker NOT redeployed** (off this path — last deploy 07-23). **No RORT_* flag touched — deliberately**: the four flags armed 07-26 were mid-validation (#68 and #12/#16 validated + closed earlier today; **#121 MTF_PB_PREV_EPOCH and #125 settle-retrue still pending their scenario**), so R left every flag exactly as found.
  - **DB migration applied to prod (22:13Z, R, direct psycopg3):** `src/migrations/task_mentions_table.sql` — new `task_mentions` table (additive, RLS on with no public policy, service-role only), unique `(comment_id, mentioned)`, inbox index `(mentioned, seen_at)`, plus the optional 7-day backfill: **39 rows** (M 21, kevin 15, F 2, R 1). Re-ran the whole file a second time to verify the idempotency claim: still 39 rows, no error. Nothing existing altered.
  - Post-deploy verification: `/api/docs` 200; `openapi.json` now serves `/api/dev-tasks/{task_id}/poll` (#148) and `/api/mentions[/seen-all|/{id}/seen]` (#143); both new endpoints return 401 unauthenticated = admin-gated as designed. Frontend `/admin/tasks` 200.
  - Observed cache gap: ~1–2 min Worker warmup gaps at ~22:07, ~22:12 and ~22:19 UTC (three container restarts, one per car). Post-close Mon (market closed 20:00Z), live-trade dormant — no live alerts lost.
  - Gates (R, all PASS): (1) commit-range per branch vs `origin/dev` = **exactly 1** for all three cars; (2) `fidelity_parity_suite.py` from RoR_Trader ROOT run **three times** — pre-train baseline `a078b13d`, post-#101 `fa282303`, post-#102 `310a8a8d` — **18/18 ALL PASS every time** (canary 267 `BAR_CACHE_ENABLED` and `RORT_RIGHTSIZE_WARMUP` ON==OFF, symdiff=0; run post-close so the RTH now-anchored false-red does not apply; the 2 SPY/10Sec fails seen earlier in the day were absent); (3) acceptance scripts `test_stamp_ticks_kevin_step_151.py` 14/14, `test_board_poll_efficiency_148.py` 14/14, `test_mentions_143.py` 23/23 — all re-run after the rebase; (4) `npx tsc --noEmit` on the rebased branch == dev-head control in the same worktree (40 == 40 pre-existing errors, zero new, none in train files); (5) GitHub `synthetic-parity` pass on every PR. Backup: **`backup/dev-pre-flane-0727`** @ `a078b13d`.
  - ⚠️ **Conflict resolved by R during the train (flagged for F):** PR #102 collided with #101 in `AdminTasksPage.tsx` `pollRefresh` (#148 had rewritten it into a throttled `(boardMax?)` form while #143 had added `loadUnread()` to the old unthrottled one) and in a `TaskDetailModal.tsx` import list. Resolution: import union; and `loadUnread()` now runs on EVERY poll tick OUTSIDE #148's throttle, while the heavy `load()`/`loadRuns()` board refetch stays throttled — because a new @-mention inserts a comment and need not move `dev_tasks.updated_at`, which would leave the unread badge stale. Cost: 1 → 2 calls per 15s tick (vs 3–4 per 7s before #148). Rebased branch head `d430e9b0`, force-pushed (that branch only).

- **20:50:30 + 20:51:03 UTC (14:50/14:51 MT)** — `6af66279` + `a078b13d` post-close docs+tooling train: PR #99 `docs/m-handoff-roster-154` (charter roster sync + **NEW §4 rule "the assignee is whoever the task is waiting on"** + the owed Deploy_Log entry for PR #97/#98) then PR #100 `docs/preflight-and-skills-153` (board #153 preflight invariant check + `/preflight` skill, bug-hunt SETTLE-CONFOUND section, dispatcher GIT CONTRACT block, V5 design doc + two M-lane specs). Merged 33s apart by **R·auto**, board #173.
  - Service(s) redeployed: api + Worker + frontend + batch-worker + Data Worker + streamlit + flat-file-cron (dev auto-deploy, watchPatterns=[] → all). Both pushes deployed: the `6af66279` set went ACTIVE then **REMOVED** ~30s later, superseded by the `a078b13d` set — all seven **SUCCESS on `a078b13d`** (api 20:51:08Z, Worker 20:51:06Z, frontend 20:51:04Z, batch-worker 20:51:07Z, Data Worker 20:51:05Z, streamlit 20:51:05Z, flat-file-cron 20:51:06Z). **shadow-worker NOT redeployed** (off this path — last deploy 07-23). No RORT_* flag change (a git push cannot alter env-var flag config).
  - Observed cache gap: ~1–2 min Worker warmup gap ~20:51–20:53 UTC. Post-close Mon (market closed 20:00Z), live-trade dormant — no live alerts lost.
  - Diff is docs + local-only tooling (`tools/preflight/`, `.claude/skills/`, `tools/team_dispatcher/` comment block, `docs/**.md`); **no engine, API or frontend code path touched**.
  - **Logged retroactively by R — release f-lane 07-27 (board #174).** R·auto merged both PRs cleanly and then died at ~20:56Z on an API connection error, before it could write this entry or report; nothing was left half-merged, but the trail stopped at the merge. This entry closes that gap, and the per-step breadcrumb protocol on #174 exists so the next mid-response death is legible instead of a mystery.

## 2026-07-26

- **04:11 UTC 2026-07-27 (22:11 MT 07-26)** — `27fa6bbb` + `a6346354` docs-sync release: PR #97 `docs/deploy-log-96` (records the #96 salvage release) + PR #98 `docs/m-lane-sync` (charter + Team_Board tombstone + Spec_Tasks_Team_Board Phase-3 amendments; board #149). Two docs-only PRs merged back-to-back (13s apart), R·auto.
  - Service(s) redeployed: api + Worker + frontend + batch-worker + Data Worker + streamlit + flat-file-cron (dev auto-deploy, watchPatterns=[] → all; Railway coalesced the two quick pushes). **shadow-worker NOT redeployed** (off this path). No functional change — docs-only diff (`.md` under `clients/.../docs/`); no code, no RORT_* flag change (a docs push cannot alter env-var flag config). Post-close Sun, market closed, live-trade dormant.
  - Observed cache gap: ~1–2 min Worker warmup gap ~04:11–04:13 UTC (container restart; off-hours, no live alerts — gap_healer backfilling).
  - **Logged retroactively by M·auto (successor), board #154 — R·auto correctly declined to self-log this release: a Deploy_Log entry for the release that carries it would need its own follow-up PR to log, an infinite regress. Resolution: the entry rides the NEXT train inside an unrelated docs PR (this one), authored by M, merged by R — not self-referential.** Bundled with the 07-27 M-handoff charter roster sync (same branch). NOT self-merged — hands to R per charter §1/§8.

- **01:38 UTC 2026-07-27 (19:38 MT 07-26)** — `9d003b2a` Merge PR #96: salvage dependency pins + nightly bug-hunt SOP (board #150; **first board-dispatched release, R·auto**)
  - Service(s) redeployed: api + Worker + frontend + batch-worker + Data Worker + streamlit + flat-file-cron (dev auto-deploy, watchPatterns=[] → all). **shadow-worker NOT redeployed** (off this path by design — commit `-`). All SUCCESS on `9d003b2a`; post-close Sun (market closed, live-trade dormant).
  - ⚠️ **DEPENDENCY-PIN CHANGE — real rebuild (the payload):** requirements.txt pins pandas `>=2.0.0`→**`==3.0.3`**, numpy `>=1.24.0`→**`==2.5.1`**, adds **`pyarrow==24.0.0`**, supabase `>=2.0.0`→**`==2.31.0`** (match the known-good 07-09 shadow-worker image; loose `>=` ranges risked build drift). Worker build logs confirm all four cp312 wheels + supabase 2.31.0 sub-deps (realtime/auth/postgrest/storage3/functions) collected & installed clean — no ResolutionImpossible/conflict.
  - Boot verification (per-service): api `Application startup complete` + `Uvicorn on :8000`, `/api/docs` **200**; frontend `/admin/tasks` **200**; Worker booted 01:39:19Z, pack registry loaded (10 general packs), pandas ResampledStore resampling live under the new pins, RORT flags read normally — no error/traceback on any service (api's two `regex`-deprecation warnings are pre-existing, non-fatal). RORT flag count unchanged (a git-only deploy cannot alter env-var flag config).
  - Observed cache gap: ~1–2 min Worker warmup gap ~01:38–01:40 UTC (container restart; off-hours weekend, no live alerts — gap_healer backfilling).
  - Gates (R·auto, all PASS): (1) commit-range `origin/dev..salvage` = **exactly 2** (`76a52117` deps, `b855c16d` nightly SOP); (2) parity suite **18/18 ALL PASS** from a clean origin/dev worktree (canary 267 symdiff=0, settled day 07-20); (3) pins == known-good shadow-worker versions; (4) code-review — **no CONFIRMED blockers** (docs+config only). Backup: `backup/dev-pre-salvage` @ `ad9a8640`.
  - Rollback (unused): `gh pr revert 96` + merge the revert normally; pins are the risk surface. No rollback required — clean.

- **~00:05 UTC (~18:05 MT 07-25)** — `5cce360c` Train 07-25 car 2: board polish + kanban lifecycle (#134/#136)
  - Service(s) redeployed: api + frontend (Worker/batch rebuild, no functional change). Post-close Sat.
  - Gates: parity 16 pass + 2 known SPY/10Sec revision-drift fails (E-lane verdicts on record ×3); npm/tsc green (F·auto, exact SHA); lifecycle migration applied pre-merge (additive).
  - Executed by M acting as R under Kevin's explicit remote authorization (recorded #134); dispatcher.py conflict resolved via F·auto rebase (#138).

- **~23:50 UTC 07-25 (~17:50 MT)** — `c3c8d6c8` Train 07-25 car 1: health nits + dispatcher hardening (#122/#131/#132)
  - Service(s) redeployed: api + frontend + Worker + batch (post-close Sat). api/docs 200, /admin/tasks 200 verified.
  - Backup: `backup/dev-pre-train-0725`.

## 2026-07-25

- **17:04 UTC (11:04 MT)** — `2685c8a0` Merge PR #79 `feat/run-button` → dev: V4.12 ·
  Registry Phase 2 (board #109) — Run button on /admin/tasks (rows + modal) that DECLARES
  a dispatch (`run-requested` tag + system comment + `run_history` row); the LOCAL
  dispatcher `--loop` EXECUTES it as a priority-jump (tag cleared on claim, lifecycle
  requested→running→ok|error|lease-expired|ignored). Task modal run-history panel
  (Config tab); /admin/agents status dropdown + prompt-template editor + per-agent
  recent runs. M review fixes in: Scoping/needs-scoping hard refusal (dispatcher +
  endpoint 400 + UI mirror) and reap() parsing from the FULL run log. New READ-ONLY
  router `run_history.py`; `POST /api/dev-tasks/{id}/run-request`; dispatcher now
  TRACKED at `tools/team_dispatcher/` (baseline + #109 delta). No RORT_* flags.
  - **Migration `run_history.sql` pre-applied to Supabase 07-25 by F2 — additive,
    RLS-on/no-policies, FK cascade to dev_tasks.** No DB change in this deploy.
  - Service(s) redeployed: api / frontend (the functional pair); Worker / batch-worker /
    Data Worker also rebuilt on the dev push (auto-deploy, no functional change) — all
    SUCCESS @ `2685c8a0`, Worker logs show normal bar_cache activity post-boot.
    shadow-worker untouched (E-lane `railway up` only). api verified: `/api/docs` 200,
    `/api/run-history` 401-gated (live route, was 404). frontend verified via headless
    Playwright (test account): Run column renders, 105 rows / 104 Run buttons, eligible
    task shows enabled ▶ Run, disabled rows carry reason tooltips; authenticated
    in-page probes 200 on dev-tasks + run-history.
  - Observed cache gap: standard ~1–2 min Worker restart window ~17:05–17:07Z (brief
    api 502 at 17:05:48Z); Saturday post-close, no live-alert impact.
  - Notes: R release session. Gates: commits = exactly the 4 expected; parity suite
    16/18 — the 2 FAILs are cache-parity SPY/10Sec 24/7 (OHLCVdiffs=43) + RTH (=1),
    the known #103/#116 revision-drift class (settled day 07-22, identical counts to
    author's run; 3rd recurrence in 4 days → V1.17), shipped on **explicit E sign-off
    board #116 comment 169** (import-graph disjoint + reproduced on clean origin/dev);
    canary-267 both legs symdiff=0. Frontend build green; tsc 40 == origin/dev 40
    (baseline moved from author's 94 after PR #78; equality holds, 0 in touched files).
    Reap full-log acceptance 9/9. Code review: no CONFIRMED blockers (nits: reap
    IndexError if task deleted mid-run; run-request double-click race can strand a
    stale ⏳ row; lease-expiry says "killed" but doesn't kill — pre-existing baseline).
    Pre-existing `synthetic-parity` CI red (missing pytest — board infra item, dev
    fails identically). Backup: `backup/dev-pre-run-button` @ `d9b7d4f7`. Merge-day
    note executed: untracked pre-fix `dispatcher.py` removed from the main checkout
    (diff-verified strictly older — exactly the two M review fixes missing); no
    dispatcher process was running; `state.json`/`logs/` left in place.

- **05:17 UTC (23:17 MT 07-24)** — `a0785db6` Merge PR #78 `feat/health-phase-a` → dev:
  Strategy Health last-10 pairing score + Phase B notes/bug-chips (board task #70,
  Phases A+B + Kevin's 4-item local-review round; approved on local, released 07-25).
  New READ-ONLY routers `health_last10.py` (last-10 backtest-trade → live-alert pairing,
  dual fired/theo basis, ±10s greedy 1:1 port of `_pair_phantom_missed`) and
  `strategy_notes.py`; `affected_sids` whitelist + validation on `dev_tasks`. Frontend:
  Last-10 score column (between Sub and Context) + sortable SID column + Context column
  (📝 notes / 🐛 affecting-task chips, replacing the old Flags chip column) + two portal
  modals (Last10Pairing, StrategyNotes). No flags; no engine paths touched.
  - **Migrations: `strategy_notes` table + `dev_tasks.affected_sids` column were
    pre-applied by M 07-24 — additive, already live, inert under old code.** No DB
    change in this deploy; nothing to roll back on the DB side.
  - Service(s) redeployed: frontend / api (the relevant pair); Worker / batch-worker /
    Data Worker / Streamlit / flat-file-cron also rebuilt on the dev push (auto-deploy).
    shadow-worker untouched (E-lane `railway up` only). api + frontend builds SUCCESS
    (images created 05:17:38Z / 05:18:53Z); api booted clean — new routes live and
    auth-gated (`/api/strategy-health-last10` → 401 unauth, not 404).
  - Observed cache gap: standard ~1–2 min Worker restart window ~05:17–05:19Z (every dev
    push restarts Worker); this change touches no Worker/engine code, so no data-logic
    impact expected.
  - Notes: gates all green on rebased `1b276a58` — fidelity_parity_suite 18/18, frontend
    build clean, code-review no CONFIRMED blockers, E semantics sign-off APPROVED (greedy
    port faithful; fired-vs-theo divergence intentional per Kevin's fired-ts ruling).
    Deploy-verified vs prod DB via the real `_score_sid` logic: 344 fired 12/20 · theo
    12/20 with blanks-not-garbage (16 blank nearest-alert cells), sid-310 M seed note
    present, sid 310 → task #3 chip. Backup: `backup/dev-pre-health-phase-a` @ `338a1fbb`.
    Non-blocking follow-ups: E's `_greedy_pair` count-by-index nit (max +1/20; fast-
    follow) and the pre-existing `synthetic-parity` CI red (workflow missing `pytest` —
    board infra item, not this PR; dev's own runs fail identically).

- **03:26 UTC (21:26 MT 07-24)** — `66ec9ea3` Merge PR #76 `feat/board-qol` → dev:
  Board QoL (board task #106, V2.7 — both rounds; Kevin approved local preview 07-24
  17:25Z, released 07-25 pre-open). Round 1: `needs-review` tag convention (amber 👀 chip,
  one-click toggle mirroring ⚡urgent, filter checkbox — agents skip tagged items, Kevin
  filters to them); sortable Flat-view column headers (ID / Pri / Task / Status-by-
  lifecycle / Area / Who, asc→desc→clear; grouped view stays pipeline-ordered); `/admin/
  tasks?task=<id>` deep links that open the task modal and keep the URL shareable via
  `history.replaceState` (no Next `useSearchParams` suspense trap). Round 2: `Scoping`
  status between Backlog and Todo in distinct purple (`#a855f7`); one-line status-
  definition tooltips VERBATIM from Session_Charters.md §7 on every status select + option;
  list rows chip ALL tags (vision, needs-scoping, …), needs-review keeping its amber
  treatment. UI only — 3 `.tsx` files (AdminTasksPage / taskBoardShared / TaskDetailModal);
  no python, no migrations, no flags.
  - Service(s) redeployed: frontend / api (the relevant pair); Worker / batch-worker /
    Data Worker / Streamlit / flat-file-cron also rebuilt on the dev push (auto-deploy) —
    all SUCCESS on `66ec9ea3` (deploys created 03:26:12–17Z). shadow-worker untouched
    (last deploy 07-23 17:02Z, E-lane `railway up` only).
  - Gates (R re-run pre-merge on `17e4641a`): exactly the 2 branch commits over dev tip
    `72216064` ✓ · fidelity parity suite 18/18 PASS, 0 FAIL — SPY 10Sec+1Min cache-parity
    green (OHLCVdiffs=0), so the #103/#116 revision-drift class did NOT recur (E's PR #77
    nightly settled-retrue held); canary-267 ON==OFF symdiff=0 (pre-open, no RTH false-
    positive) ✓ · frontend `npm install && npm run build` exit 0 ✓ · code review: 0 confirmed
    blockers (2 non-blocking minors — sort `<th onClick>` headers lack keyboard a11y,
    consistent with existing clickable toggle spans; deep-link to a nonexistent task id
    relies on the modal's pre-existing graceful handling) ✓. GitHub "Fidelity Gate" red =
    the same pre-existing missing-`pytest` CI infra failure as every dev push (the
    synthetic-parity suite itself passed 31/31; board backlog "CI: add pytest to the
    synthetic-parity workflow").
  - Deploy watch: frontend SUCCESS 03:26:13Z, api SUCCESS 03:26:17Z — both boot clean (api
    `/health` 200, auth routes 401 = routing live; frontend `/admin/tasks` 200). Deployed
    bundle confirmed to carry both rounds: served chunks `3931.*` / `4877.*` contain
    `Scoping`, `#a855f7`, `needs-review`, the verbatim §7 tooltips, the 👀 toggle, and the
    `?task=` deep-link (`replaceState` / `searchParams`). Interactive click-through not run
    — no browser/Playwright MCP in this R session; served-code-presence proof used instead.
    Observed cache gap: Worker restart window ~03:26–28Z, outside RTH (07-25 pre-open) —
    none expected.
  - Backup branch: `backup/dev-pre-board-qol` (`72216064`). Rollback = `git revert -m 1
    66ec9ea3` (UI only — no DB / flags to roll back).

## 2026-07-23

- **20:41 UTC (14:41 MT)** — `9a45898` Merge PR #74 `feat/agents-registry` → dev:
  Spec_Agents_Registry.md Phase 1 (board #99, V2.6) — additive `agents` table + touch
  trigger + RLS + idempotent 7-row seed (scope/boundaries verbatim from charter §1;
  charter stays SSOT until the V4.9 dispatcher inverts authority), agents CRUD router
  (fail-loud enums) + main.py registration, read-only /admin/agents department-grouped
  roster cards (letter avatars, status chips, expandable scope/boundaries, worktrees,
  context docs, live queue from the board), registry-driven assignee/comment-author
  dropdowns with the ASSIGNEES const as fallback. Tooling only (no trading tables, no
  engine files); no flags. **agents migration+seed pre-applied 07-23** (additive; inert
  under old code — deploy order was a non-issue).
  - Service(s) redeployed: api / frontend (the touched pair); Worker / batch-worker /
    Data Worker also rebuilt on the dev push (auto-deploy) — all 5 SUCCESS on `9a45898`
    (deploys created 20:41:02–05Z). shadow-worker untouched (E-lane `railway up` only).
  - Gates (R re-run on `cda2f3d` pre-merge): only the 1 branch commit over dev ✓ ·
    fidelity parity suite 18/18 PASS, 0 FAIL — SPY/10Sec cache-parity green, so the
    board #103 revision-drift class did NOT recur (transient gate RED during the build
    resolved as #103: bar_cache revision drift on SPY 07-20, Polygon post-capture
    revision; E re-trued and closed ~20:00Z; unrelated to this branch) ✓ · frontend
    production build exit 0 ✓ · code review: 0 confirmed blockers (2 minors, both
    unreachable under enum validation: duplicate-letter POST surfaces as raw 500 not
    clean 400; unknown-department cards wouldn't render) ✓. GitHub "Fidelity Gate" red =
    same pre-existing missing-`pytest` CI infra failure as every dev push (board backlog
    item "CI: add pytest to the synthetic-parity workflow").
  - Deploy watch: API SUCCESS 20:41:40Z, boot clean (startup complete; only the two
    pre-existing FastAPI `regex=` deprecation warnings); frontend SUCCESS 20:43:23Z.
    Playwright spot-checks on the deployed pages: /admin/agents renders all 7 roster
    cards grouped under DEV with live queue counts (F card "11 open" = API count);
    sidebar shows Agents next to Tasks; tasks-page assignee select loads registry
    letters incl. E2 — E2 exists only in the DB, proving the registry round-trip is
    live, not the const fallback. Timing precondition honored: merged 20:41Z (post
    20:00Z close) with board #100 re-confirmed Done in the DB. Observed cache gap:
    Worker restart window ~20:41–43Z, outside RTH — none expected.
  - Backup branch: `backup/dev-pre-agents-registry` (fde9d28). Rollback = `git revert
    -m 1 9a45898` (agents table additive — no DB rollback needed).

- **04:08 UTC (22:08 MT 07-22)** — `ed4a0b6` Merge PR #72 `feat/task-detail-panels` → dev:
  Spec_Tasks_Team_Board.md Phase 3 + 07-22 amendments (board task #84, V2.5) — three-panel
  task modal (Context tabs re-scope to the selected pipeline item, Activity thread, vision
  pipeline strip), subtask carets, handoff-pipeline checklists driving the Next-actor chip,
  circular role avatars + avatar-click pickers replacing the @ dropdowns. Tooling only
  (dev_tasks router / task views / md-render deps); no trading tables, no engine files, no
  flags. **checklist migration pre-applied 07-22** (additive; inert under old code); Kevin
  sign-off on local 07-22 (four review rounds).
  - Service(s) redeployed: api / frontend (the touched pair); Worker / batch-worker /
    Data Worker also rebuilt on the dev push (auto-deploy) — all 5 SUCCESS on `ed4a0b6`
    (deploys created 04:08:33Z). shadow-worker untouched (E-lane `railway up` only).
  - Gates (R re-run on `7a75c15` pre-merge): only the 5 branch commits over dev ✓ ·
    fidelity parity suite 18/18 PASS, exit 0 ✓ · frontend production build exit 0 ✓ ·
    code review: 0 confirmed blockers (2 minors: unsaved-description draft discarded on
    pipeline-selection switch without dirty-check; duplicate comment on held Enter —
    pre-existing pattern) ✓. GitHub "Fidelity Gate" red = same pre-existing
    missing-`pytest` CI infra failure as every dev push since ≥07-16.
  - Deploy watch: API switchover blip 04:24:24Z (single 000, then healthy). Playwright
    spot-checks on deployed /admin/tasks 10/10 PASS: Next chips render; vision modal
    above sidebar (full-viewport fixed overlay z-1000); pipeline-strip selection
    re-scopes Context tabs + Activity header to the selected subtask; avatar-click
    role-picker popover (8 role circles); PATCH invalid-`checklist` → new 400 validation
    message (new API code confirmed live); 28 legacy 'claude' rows intact via API.
    Worker boot clean (warmups serving from ResampledStore, no errors). Observed cache
    gap: none expected (overnight, no session).
  - Backup branch: `backup/dev-pre-task-detail-panels`. Rollback = `git revert -m 1
    ed4a0b6` (checklist column additive — no DB rollback needed).

- **03:12 UTC (21:12 MT 07-22)** — `ef447bd` fix(engine): BAR_DUP_GUARD — duplicate-period
  rows / bar-count inflation in BarBuilder (PR #73, flag `RORT_BAR_DUP_GUARD` default OFF)
  - Service(s) redeployed: Worker, api, batch-worker, frontend (dev auto-deploy; **flag OFF —
    inert**; shadow-worker untouched by design)
  - Found by the 07-23 nightly bug-hunt: late out-of-order tick re-opens an already-closed
    period → duplicate history row + `_bar_count` over-increment → 136's max-hold exits fire
    one bar early; dup row also feeds incremental indicators + a spurious bar-close dispatch.
  - Always-on `BAR_DUP_GUARD` tripwire WARN (rate-limited 1/period/builder) ships with this —
    today's RTH logs quantify the class BEFORE any arm. **Arm held for Kevin** (nightly rail 2):
    `railway variables --set "RORT_BAR_DUP_GUARD=1" --service Worker`; revert = set `0`.
  - Validation: new `test_ralph_bar_dup_guard.py` 4/4; ralph suites 4/4+6/6+23/23;
    fidelity_parity_suite 18/18 AND full `--coarse --writethrough` 29/29 — each flag OFF and ON.
  - Boot: one transient "Server disconnected" engine crash 03:14:15 during switchover
    (supervisor restarted; warmups clean from 03:14:36). Observed cache gap: none expected
    (overnight, no session).
  - Also folds in board V4.6 (`.claude/skills/session-handoff/` now tracked in git).

## 2026-07-22

- **22:37 UTC (16:37 MT)** — `bf0d2cf` Merge PR #71 `feat/tasks-team-board` → dev:
  /admin/tasks becomes the team board (Spec_Tasks_Team_Board.md Phase 1, board V2.4) —
  vision/subtask hierarchy, role assignees (M/E/E2/F/P/R/kevin), `?assignee=` poll filter.
  Tooling only (dev_tasks router / AdminTasksPage / additive migration); no trading
  tables, no engine files, no flags. **Migration was pre-applied 07-22** (additive;
  51 existing rows verified untouched), so deploy order was a non-issue.
  - Service(s) redeployed: api / frontend (the touched pair); Worker / batch-worker /
    Data Worker also rebuilt on the dev push (auto-deploy) — all 5 SUCCESS on `bf0d2cf`.
    shadow-worker untouched (E-lane `railway up` only).
  - Gates (R re-run pre-merge): only `f188b83` over dev ✓ · fidelity parity suite
    18/18 PASS, exit 0 ✓ · frontend production build exit 0 ✓ · code review: no
    confirmed blockers ✓. (GitHub "Fidelity Gate" check red = pre-existing
    missing-`pytest` infra failure on every dev push since ≥07-16; the 31 CI parity
    tests themselves pass before the import dies — not a regression of this PR.)
  - Deploy watch: API + frontend SUCCESS by ~22:40Z; live verify = login → GET
    `/api/dev-tasks` (51 rows, `parent_id`/`origin` on all, `?assignee=` filter exact,
    28 legacy 'claude' rows intact) + deployed /admin/tasks chunk serves the new
    "By vision" view. Backup branch: `backup/dev-pre-tasks-team-board`.
  - Observed cache gap: post-close push (22:37Z) — usual ~1–2 min Worker warmup only,
    no RTH impact.

- **02:05–02:22 UTC (20:05–20:22 MT 07-21)** — **shadow-worker `railway up` from the MERGED
  tree** (`merge/mrs5a-plus-dev-0722` @ff1427b = mrs5a Ph1/2b/2c + dev's submin-canonical
  stack; only conflict = services.py secondary block, resolved by moving the Phase-2b
  sub-minute hook INSIDE `compute_secondary_columns` so the resident frame shares it).
  Gates before deploy: 30/30 unit, parity 16/16 (one SPY/10Sec transient isolated as
  settle-sweep revision-edge — cache-path files byte-identical to dev), Step-C manager
  byte-identity 263 (162==162) + 267 (103==103). FINGERPRINT SOP: up-boot printed
  `code fingerprint=cf237aca9f39` == upload-tree hash ✓; then
  `RORT_CANONICAL_SUBMIN_STATE=1` var-set (rebuild verified as the pinned upload via the
  M-RS5a-only "RESIDENT lane active" boot line + digest exclusion of the old images).
  ⚠️ Lesson: `railway up` must run from the REPO ROOT (service rootDirectory appends
  `/clients/KevBot_Toolkit/RoR_Trader`); the first attempt from the subdir FAILED.
  - **RESULT: shadow_heartbeats UNFROZE on the first pass** (02:07-08Z — all 4 canaries
    caught up to 07-21 19:59:50 close; frozen since 07-20 19:19Z = wedged process, cleared
    by the restart). Intraday board freshness restored for the canary SIDS set.
  - Also tonight: `RORT_RECOMPUTE_PARALLELISM=6` on batch-worker (345 ×8-pool crash
    mitigation; revert = same command with 8). Nightly bug-hunt brief prepended to
    `Divergence_Hunt_Log.md` (nothing armed by the hunt).

## 2026-07-21

- **~21:07 UTC (15:07 MT)** — submin-canonical FIX #2 (`22c6b1f`): the (10,'RTH') shadow
  was never CREATED live — 267/338 are utv4-triggered, so interp-aware suppression said
  "real monitor covers UT_BOT_V4"; 339's gate had consumed 267's incremental own-records
  all along. Under the flag, sub-minute keys now always get a dedicated canonical-owner
  shadow. **VERIFIED LIVE 21:12Z**: 10s shadow warmed (11,700 bars) +
  `[canonical-submin n=3000]` publishes flowing, 0 fail-loud errors. (Also 20:59Z:
  `RORT_SUBMIN_DERIVE_BARS=3000` set explicitly on Worker — default value; documents the
  knob + triggered the diagnostic restart.)
  - Service(s) redeployed: Worker / api / batch-worker / Data Worker
- **~20:5x UTC (14:5x MT)** — submin-canonical FIX push: the (10,'RTH') shadow on the LIVE
  TSLA hub closes via `_close_shadow_with_bar` (10s is ALSO a primary — 267/338 — and the
  per-second secondary loop skips primary TFs), which the canonical branch didn't cover —
  live stayed legacy-incremental while the single-monitor harness showed canonical (the
  label-drift-class topology trap, caught by post-arm log verification). Derive factored
  into `_canonical_submin_close`, owned by BOTH close sites; +3 topology regression tests
  (22/22). Also: harness ARMED_FLAGS mirror (+RORT_CANONICAL_SUBMIN_STATE=1).
  - Service(s) redeployed: Worker / api / batch-worker / Data Worker
- **20:26 UTC (14:26 MT)** — **ARMED `RORT_CANONICAL_SUBMIN_STATE=1`** on Worker + api +
  batch-worker + Data Worker (Kevin-directed; Claude flipped via railway CLI after
  permission grant). Store seeded first: TSLA 10Sec/30Sec × RTH/Extended, 130d, 837,237
  rows, 76 chunks, 0 comparator diffs. Gate 1 = 18/18 flag OFF AND ON (267: 301==301,
  symdiff 0). Harness matrix: 314/340/267 ceilings unchanged ON vs OFF (92.3/100/100).
  ⚠️ 339's backtest REBUILDS on true-10s bars at next recompute (by design).
  NOT armed: shadow-worker (pinned snapshot; needs `railway up` + fingerprint SOP; its
  4-sid set has no sub-minute-gated strategy, so no lane divergence while held).
- **17:36 UTC (11:36 MT)** — PR `feat/submin-canonical-source` → dev: sub-minute canonical
  bar+state source (Phase 2b, `Plan_SubMinute_Canonical_Source.md`). **Flag-OFF deploy —
  `RORT_CANONICAL_SUBMIN_STATE` default OFF = byte-identical everywhere** (unit-gated:
  19/19 canonical-state tests + 70/71 sweep, 1 pre-existing fail). Includes the harness
  secondary-TF-gap port onto dev (GEN packs / sub-minute re-true / fail-loud UNRES).
  - Service(s) redeployed: Worker / api / batch-worker / data-worker (dev auto-deploy)
  - Observed cache gap: expect the usual ~1-2 min Worker warmup gap; mid-RTH push
    (Kevin-approved "push and confirm").
  - Notes: ARMING deferred to the post-close gate sequence (parity suite OFF/ON, harness
    matrix, seed) — see `Impl_SubMinute_Canonical_Source.md`. ⚠️ When armed, the flag
    CHANGES sub-minute-gated backtests (339) on next recompute — by design (one canonical
    bar, both lanes). Kill switch: flip the one flag OFF.

## 2026-07-16

- **~19:10 UTC (13:10 MT)** — M-RS5b push + arm (late-RTH, Kevin-approved aggressive tempo):
  `RORT_CANONICAL_FINE_TF_STATE` — fine RTH gate state rebuilt from the canonical resample
  of the hub's own 1Min history at EVERY close (one construction path; fail-loud, no silent
  fallback). Armed together with `RORT_MTF_STATE_REFRESH_S=0` — **the 120s refresher is
  RETIRED** (Kevin: full transition, loud failures). A/B acceptance GREEN: 329 0→1 exact,
  328 7→8/8 second-exact, 327 4/5→5/5, no regressions (see Plan_MRS5_2026-07-16.md).
  **⚠️ bar-gap windows: this push + arm ≈ 19:10-19:25Z** (last ~40 min of RTH — noted for
  divergence reads; EH provides the clean 5b observation window).
  - Service(s) redeployed: Worker + api (twice: push + var-set arm)
  - Reversible: `--set RORT_CANONICAL_FINE_TF_STATE=0 --set RORT_MTF_STATE_REFRESH_S=120`

- **~16:52 UTC (10:52 MT)** — batched push (MID-RTH, Kevin-approved; trading dormant):
  P1-hot-reload (`RORT_HOTRELOAD_BOOT_PARITY`, worker.py boot-parity warmup + edit-path
  shadow finalize) + P2 bounded-boot healthcheck (`RORT_HEALTHCHECK_BOUNDED_BOOT`,
  engine_health.py + write-once boot marker in worker.main; deadline default 1200s) +
  smoke-canary historical window (local_update.py — validated live 16:46Z: hist 2014/2014
  bit-perfect, lane_only=0 during RTH). Both new flags shipped **OFF** (byte-identical
  baseline; 13 companion tests + Brandon's tests flag-flip-verified). Then ONE combined
  arm (both flags in one var-set). **⚠️ For divergence analysis: bar-gap windows today =
  this push + the arm (~16:52-17:05Z) + stalls 15:16:23Z (20.8s, cause unknown) and
  16:48:18Z (15.3s — correlated with local smoke-recompute DB load, not flags).**
  - Service(s) redeployed: Worker + api
  - Reversible: `--set RORT_HOTRELOAD_BOOT_PARITY=0` / `--set RORT_HEALTHCHECK_BOUNDED_BOOT=0`

## 2026-07-15

- **~22:15 UTC (16:15 MT)** — grace fix push: `RORT_GRACE_FINAL_CLOSE_ELIGIBLE`
  (Brandon P1, no-signal-at-grace). Shipped **flag OFF** (byte-identical baseline). Then
  ARMED overnight (var-set) so tomorrow's open runs it — narrow: only sub-minute
  ws_rest_spliced/ws_agg_reconciled strategies whose grace emits no signal but the final
  second forms one. All affected are RTH (no EH effect tonight); arming now avoids an
  open-gap. Reversible: `railway variables --service Worker --set RORT_GRACE_FINAL_CLOSE_ELIGIBLE=0`.
  - Service(s) redeployed: Worker (twice — code push + var-set arm)
  - Observed cache gap: RTH closed / EH — no live bars to gap
- **~21:58 UTC (15:58 MT)** — ARMED `RORT_CANONICAL_PRIMARY_CLOSE=1` on Worker (var-set
  redeploy). Blast radius 8 live 1Min+ default-model strategies incl. 340. Worker came up
  SUCCESS + healthy (engine warming normally, no stall/crash). Reversible:
  `railway variables --service Worker --set RORT_CANONICAL_PRIMARY_CLOSE=0`.
- **~21:23 UTC (15:23 MT)** — `206002b` feat(engine): RORT_CANONICAL_PRIMARY_CLOSE — `>=60s`
  primary close/dispatch fix (Brandon audit P0+P1). **Flag default OFF → NO trading-path
  behavior change on this deploy** — flag-OFF is byte-identical to baseline (stash-verified
  twice). Single-sources the ws_agg-consumer set so a default-model (`ws_rest_spliced`) `>=60s`
  primary gets canonical A-sub/1Min-dispatch/fan-out instead of an incomplete flush partial;
  flush no longer drives `>=60s` strategy closes on partials. Bundles `7c1b651` (harness
  pairing/windowing fixes), `5c06682` (Brandon audit doc+tests, cherry-picked), and the
  `/replay-check` skill.
  - Service(s) redeployed: Worker + api
  - Observed cache gap: RTH closed (post-20:00Z) — no live bars to gap
  - Notes: flag OFF; arming `RORT_CANONICAL_PRIMARY_CLOSE=1` on a single default-model 1Min+
    canary is a later, announced, monitored step. Nightly 00:20Z is >3h out.

## 2026-07-14

- **~20:55 UTC (14:55 MT)** — `5fdaeaa` (PR #66) feat(ops): engine liveness watchdogs.
  `[ENGINE-STALL]` (periodic-loop lateness > 15s), `[ALERT-LAG]` (dispatch > 30s after its
  bar closed), and a HEALTHCHECK that **ages out the ENGINE heartbeat** instead of testing
  file existence (the old check watched a file touched by the MANAGER thread — a starved
  engine passed it forever; healthcheck weakness also flagged by the external Sol audit).
  Observability only — no trading-path behavior change. Tests 8/8 incl. the exact regression
  (manager fresh + engine stalled → UNHEALTHY); suites at baseline; parity --quick 16/16.
  Post-deploy: watchdogs SILENT (0 stalls / 0 lag warnings / 0 errors) and alert dispatch lag
  median **6s** (max 8s) — i.e. the engine is healthy with resync off, and any future stall
  now announces itself within seconds. Backup: `backup/dev-pre-liveness-2026-07-14`.
- **🚨 20:11 UTC (14:11 MT) — `RORT_PRIMARY_STATE_RESYNC_S=0` + `RORT_PRIMARY_STATE_RESYNC_APPLY=0`
  (DISABLED) on Worker.** ROOT CAUSE of today's latency + missed entries: each resync cycle
  full-warmup-replays ~50 strategies SEQUENTIALLY (~4s each → **~9.5-min cycle**, 18:39:32→
  18:49:01) holding the GIL, repeating every 900s → the live engine was starved most of the
  time. Damage: **fleet-wide alert dispatch lag 5–10 min** (fill 18:54:30 → saved 18:59:43,
  ~13 sids; fill 19:06:00 → saved 19:16:41) and **MISSED live entries** — sid 333's 18:44:30
  entry exists in BOTH backtest and algo lanes with gates open and byte-clean primary bars
  (three-lane: algo==backtest, live absent → live-path stall). **Verified recovery after
  disable: median alert lag 4s (was 320s+), BAR_CLOSE cadence normal.** Feature (the 340-class
  primary-drift healer) needs redesign before re-arming — shared engine per (tf,session,
  indicator-signature), chunked cycles, wall-clock cap. Memory:
  `project_primary_resync_engine_starvation`. **NOTE: this CORRECTS the 19:20Z entry below —
  the incident was NOT primarily caused by local DB load; the stall predates it.**
- **⚠️ 19:20–19:33 UTC — OPS INCIDENT (originally mis-attributed to local DB load; see 20:11Z
  entry above — the real cause was the PRIMARY-RESYNC stall):** Supabase broken-pipe
  burst on Worker (78 lines, 19:20–19:22Z) incl. **14 ALERT SAVE FAILED** (canary/PACKTEST
  sids — those alerts are LOST), plus **alert dispatch lag of 10–12 minutes** (fill_ts
  19:06:00 → saved 19:16:41; 19:21:30 → 19:33:20; normal is 3–5s). Cause: heavy LOCAL
  analysis against the same prod Supabase during RTH (retro-replay 3d bar loads + two event
  audits + `_measure_five` runs + a `local_update` smoke recompute, 19:00–19:25Z). Worker
  never died — bars + alerts normal again by 19:41Z. **Consequence: the 19:06–19:33Z slice of
  today's post-arm measurement is CONTAMINATED** (delayed/lost alerts read as missed).
  Rule adopted (memory `feedback_local_analysis_starves_live_worker`): keep local prod-DB
  load light during live sessions; defer replays / local-update / fleet scans to after close;
  verify alert `timestamp − fill_ts` lag (normal 3–8s) before trusting any paired-% window.
- **~17:50 UTC (11:50 MT)** — PR #65 merged (`feat/shadow-retrue-force-full`) +
  `RORT_SHADOW_RETRUE_FORCE_FULL=1` armed on Worker right after deploy SUCCESS.
  ROOT-CAUSE fix for the residual class: shadow re-trues (refresher / session reloads /
  coarse reloads) were hitting `recompute_from_history`'s snapshot fast-path (NO alignment
  check) — restoring the live lineage instead of replaying clean, so drifted states survived
  every re-true (proven live: TSLA 5m SWING NEUTRAL 14:35→15:06+ across refresher cycles vs
  canonical BULL_C2 from the identical df; the 14:39 bt-only misses on 327/328). Flag ON =
  force full replay on re-true paths. Kill-switch = unset. Backup branch
  `backup/dev-pre-retrue-forcefull-2026-07-14`. Validation: new lock suite 3/3, ralph
  13+1 baseline, gap-heal 23/23, prior lock suites green, parity --quick 16/16.
  Watch: post-arm burst of `[MTF-REFRESH] records changed` (lineage re-truing) then
  convergence; refresher cycle duration (full-replay cost, thread-side).
  Kevin ruling in effect: not actively trading — RTH deploys OK (memory
  `feedback_deploy_autonomy` update).
- **15:40 UTC (09:40 MT)** — `<this commit>` docs: overnight-loop close-out (goal check +
  residual-class finding). Also retro-logs the **11:19 UTC** docs-only push (`e9e3835`, T3
  battery results) — both are docs-only redeploys, no code/flag changes.
- **02:38 UTC (20:38 MT 07-13)** — `RORT_WARMUP_PREV_CHAIN=1` +
  `RORT_MTF_FINE_INCREMENTAL_AUTHORITY=1` + `RORT_MTF_STATE_REFRESH_S=120` (600→120) set on
  Worker in one var-set (single redeploy). Claude-armed per standing authorization (overnight
  loop). Both flags ship in the merges below; kill-switch = unset. Boot watch:
  `[FINE-DUP-SKIP]` (first duplicate per TF), `[MTF-SEED]`, refresher cadence now 120s.
- **02:36 UTC (20:36 MT 07-13)** — `c1e8875` (PR #61) + `eb4ae5b` (PR #62) merged back-to-back
  after nightly recompute went quiet (69 refreshed, 12+ min quiet at 02:35Z).
  - PR #61 `RORT_WARMUP_PREV_CHAIN`: warmup/recompute populate the prev-value chain — fixes
    shift-based pack interpreters (MACD_HISTOGRAM_V2) emitting NO records on all derive paths
    (sid 285 0/204 events; sid 136 15m key EMPTY 5–20 min post-boot). Also documents the
    RETRACTION of the 07-13 "MTF publish clobber" (cross-hub telemetry aggregation artifact).
  - PR #62 `RORT_MTF_FINE_INCREMENTAL_AUTHORITY`: stops the fan-out duplicate branch from
    re-poisoning fine-TF shadow gate states every ~2-3 min from shallow noisy history — the
    proven state-divergence engine behind 327/328/329's 5m/2m SWING gate unpairing (GATE_DIAG
    18:41:03 C2 fire == settled truth, vs telemetry steady C3; cascade re-poison 11s after the
    18:42:00 boundary).
  - Service(s) redeployed: all dev services on merge (shadow-worker excluded, pinned). Worker
    deployment SUCCESS 02:36:21Z on `eb4ae5b9`, then var-set redeploy ~02:38Z.
  - Backup branch: `backup/dev-pre-prevchain-fineauth-2026-07-14`.
  - Validation (both PRs, pre-merge): new lock tests 5/5 + 4/4; test_ralph_fidelity 13+1 known
    pre-existing (== clean-dev baseline); test_ralph_gap_heal 23/23; fidelity_parity_suite
    FULL 18/18 PASS (incl. both previously-failing SPY/10Sec legs) + --quick 16/16 on the #62
    tree. Flag-OFF byte-identity locked by tests in both PRs.

## 2026-07-13

- **~21:45 UTC (15:45 MT)** — `RORT_PRIMARY_STATE_RESYNC_APPLY=1` set on Worker (no code
  change; var-set redeploy). Claude-armed per standing authorization + Kevin's "everything
  firing by tomorrow" directive. Swap path is replay-proven (no-drift swap == no-op,
  byte-identical; alignment-gated; grace-fire-suppressed). Overnight = guaranteed no-op;
  tomorrow intraday drift heals within one 900s cycle. Watch `[PRIMARY-RESYNC] sid=…
  APPLIED` lines. Rollback = unset the var (meter keeps running via RESYNC_S).

- **21:18 UTC (15:18 MT)** — `f3b297b` (PR #60) feat(fidelity): primary-TF pack state re-sync
  (the 340-class fix) — default OFF
  - Service(s) redeployed: all dev services on push to dev (shadow-worker excluded, pinned).
    Worker deployment SUCCESS 21:18:36Z.
  - **~21:14 UTC: `RORT_PRIMARY_STATE_RESYNC_S=900` set on Worker** (dry-run drift METER only;
    `RORT_PRIMARY_STATE_RESYNC_APPLY` deliberately NOT set — verify-first). Claude-armed per
    standing authorization. Expect `[PRIMARY-RESYNC-DRY]` drift lines each 900s cycle; review
    fleet drift before flipping APPLY.
  - Worker re-warm gap expected ~21:18–21:25 UTC.

- **20:56 UTC (14:56 MT)** — `d48c884` (PR #59) feat(M-RS2-P2 Phase 4): live-lane store serve
  (`_load_warmup_df` serves settled history from the resampled store) + weekend-head coverage
  tolerance fix in the shared splice
  - Service(s) redeployed: all dev services on push to dev (shadow-worker excluded, pinned).
  - **20:59 UTC: `RORT_RESAMPLED_STORE_SERVE_LIVE=1` set on Worker** (second redeploy) —
    Claude-armed per Kevin's standing authorization (2026-07-13). Rollback = delete the var.
  - Expected on Worker boot: `[ResampledStore#4-live] SERVED` per store-TF warmup; any
    `serve failed`/fallback lines are safe (deep path == old behavior).
  - Worker re-warm gap expected ~20:57–21:05 UTC (two restarts) — filter from divergence reads.
  - NOTE: earlier 07-13 armings (15:35 PB_DEFER, 15:41 GATE_FAIL_CLOSED, 16:42 STORE_SERVE
    offline, 17:07 MTF_STATE_REFRESH_S=600, 18:25 maintain 900s) are recorded in
    `docs/_active/Plan_MRS2_P2_Rollout.md` execution log.
  - Backup of pre-merge dev: branch `dev-backup-pre-serve-live`.

## 2026-06-29

- **~17:55 UTC (11:55 MT 06-29)** — `2010063` (PR #6) feat(bar_cache): write-through unification step 2 + `9595387` (PR #5) read_bars autocommit bar-leak fix
  - Service(s) redeployed: ALL dev services on push to dev — api / Worker / **Data Worker** /
    Streamlit / frontend / flat-file-cron / batch-worker. Data Worker confirmed `2010063` SUCCESS.
  - **Write-through gated default-OFF** (`RORT_BARCACHE_WRITETHROUGH` unset on Data Worker → off in code).
    No live behavior change; ships the code so Kevin can flip the flag for the live RTH burn-in.
  - Bar-leak fix (#7 idle-in-txn strand) now live via PR #5. Migration `bar_cache.settled`+`revised_at`
    was already applied to prod in a prior session (metadata-only).
  - Worker re-warm on restart → expect ~1–8 min gap from ~17:56 UTC; filter from divergence readings.
  - Backup of pre-merge dev: branch `backup/dev-pre-writethrough-0629` (origin/dev @ bc9cf21).

- **~17:5x UTC (11:5x MT 06-29)** — FLAG FLIP: set `RORT_BARCACHE_WRITETHROUGH=1` on **Data Worker**
  (env-triggered redeploy, dep `938ec23a`). Live RTH burn-in begins. Flag read per-call at runtime,
  so the redeploy is what activates it. Instant rollback = unset the var (forces another redeploy).
  Watching: bar_cache write-through writes (settled/revised_at), Data Worker logs for write-through
  warnings, and the ~1–8 min restart gap on this service only (alert Worker unaffected).
  - **~18:31 UTC FOLLOW-UP FIX (DB, not deploy):** burn-in monitoring caught `settle_sweep` 57014
    statement-timeouts on TSLA/SPY/TSLL — the sweep UPDATE (`symbol=X AND settled=false AND ts<cut`)
    had no index on `settled` (only PK), so it scanned millions of rows. Added partial index
    `idx_bar_cache_unsettled ON bar_cache(symbol, ts) WHERE settled=false` via CREATE INDEX
    CONCURRENTLY (non-blocking, 64s). Migration: `src/migrations/bar_cache_unsettled_index.sql`.
    After: sweep plan = Index Scan; drained 1532-row backlog in 2.4s; failures stopped 18:33:15;
    unsettled bounded ~3k with 0 stuck past the 16-min window. Data correct throughout (flag lag only).

## 2026-06-26

- **~late UTC 06-26** — `4a22777` fix(recompute): preserve-range guard (UAD no longer truncates to data_days)
  - Service(s) redeployed: api / Worker / frontend (push to dev)
  - Kill-switch `RORT_UAD_PRESERVE_RANGE` (default ON). Isolated to `recompute_and_persist`;
    fidelity parity suite 18/18 green (unaffected — suite reads via get_strategy_trades). Validated
    read-only sid 265: 122→280 trades (range preserved 06-11..06-26 vs truncated 06-22..06-26).
  - Weekend / markets closed → Worker re-warm is harmless. NEXT: safe to do the fleet-wide
    "Update All Data on everything" pass once this is confirmed deployed (no more truncation).

- **~23:31 UTC (17:31 MT 06-26)** — `1e4daa0` chore: re-trigger deploy of b47bbbc (coarse-secondary fix)
  - Service(s) redeployed: api / Worker / frontend (push to dev)
  - **WHY:** the `b47bbbc` push (~22:xx) FAILED to deploy on a **Railway deploy-infra incident**
    (prisma deploymentSnapshot.upsert FATAL: server login failing; "deployments slow to go out"
    banner). Build succeeded — not a code issue. Empty re-trigger commit deployed cleanly once
    the incident eased. api confirmed: fresh Uvicorn startup, no import errors.
  - **VALIDATED:** `RORT_COARSE_SECONDARY_FROM_1MIN=1` (api) + b47bbbc → sid 338 "Update All Data"
    completed in **~76s / 817 trades** (job 105732732) — previously hung 17+ min. Coarse UAD fixed.
  - Worker restart ~23:39 → ~8-min warmup (filter ~23:39–23:48 UTC from divergence readings).

- **~22:xx UTC (16:xx MT 06-26)** — `b47bbbc` fix(backtest): build coarse (>=1H) secondary gate from 1Min (FAILED deploy — Railway incident; superseded by 1e4daa0)
  - Service(s) redeployed: Worker / api / frontend (push to dev)
  - **INERT** — both kill-switches default OFF (`RORT_COARSE_SECONDARY_FROM_1MIN`, `RORT_COARSE_LAYER_READ`).
    No behavior change until enabled. Fidelity parity suite green (18/18, flags-off baseline).
  - Notes: fixes the coarse-gate UAD hang (sid 338) when enabled. Worker restart = ~8-min warmup + brief
    338-canary interruption. After-hours.

- **~20:30 UTC (14:30 MT 06-26)** — Worker **restart only** (no code change; `railway redeploy`)
  - Service(s) redeployed: Worker
  - Purpose: re-run `_warmup_all` to seed coarse-TF gates after-hours + warm new canary sid 338.
  - ✅ **Verified the coarse-gate fix:** boot logs `[MTF-SEED] TSLA tf=86400s (5 records)`,
    `tf=14400s (3)`, `tf=3600s (2)` — 1Day/4H/1H gate dicts now POPULATED at warmup (were empty).
    `Warmup TSLA: strat=338 tf=10s bars=11700 initialized=True`.
  - After-hours → minimal divergence impact; no filtering needed for live readings.

- **~18:5x UTC (12:5x MT 06-26)** — `a38f876` fix(live): seed cross-TF gate from warmed shadow (coarse 1H/4H/1D gates fire live)
  - Service(s) redeployed: Worker / api / frontend (push to dev)
  - ⚠️ **Deployed DURING live hours** (RTH) — Worker restart = ~1-2 min live_bars gap +
    full re-warm. **Filter ~18:5x–19:0x UTC out of divergence readings.** Intentional
    (Kevin wants to confirm coarse-gated strategies fire live this session, not wait).
  - Notes: behavior change — strategies with ≥1Hour confluence gates (e.g. 310/313) should
    now START firing live (were silent). Kill-switch `RORT_SEED_MTF_FROM_WARMUP=0` if it
    misbehaves. Watch worker logs for `[MTF-SEED] tf=86400` + 310/313 alerts.

## 2026-06-25

- **~04:0x UTC (22:0x MT 06-25)** — `cd6056c` Merge M-RS2 Phase 2 (REST Bars read path + Hi-Fi load-once)
  - Service(s) redeployed: Worker / api / frontend (push to dev)
  - Observed cache gap: **deployed while markets CLOSED** → no live-capture/divergence
    disruption expected. New dep `psycopg[binary]` installs on this build.
  - Notes: **Code is INERT until env set** — `BAR_CACHE_ENABLED` off + no
    `SUPABASE_CONNECTION_STRING` on Railway = falls back to Polygon, zero behavior change.
    To activate: set `SUPABASE_CONNECTION_STRING` (session-pooler DSN) on **api + worker**,
    then `BAR_CACHE_ENABLED=1`. Backup branch `dev-backup-2026-06-25-pre-mrs2-p2`. Once on,
    backtest recompute reads REST Bars (1.8×) and Hi-Fi reads load-once 1s (4.26×), both
    validated byte-identical. Watch first Update-All tomorrow for the speedup.
- **~01:52 UTC (19:52 MT 06-25)** — `fe666a4` tasks ID column + EOD docs/corrections (no engine/worker code change)
  - Service(s) redeployed: frontend (ID column) / api / Worker (restart on push, no code delta)
  - Observed cache gap: (deploy window — expect ~1-2 min Worker restart gap)
  - Notes: Docs + frontend only. M-RS2 Phase 2 (read_bars) stays on feat/m-rs2-phase2-readpath,
    NOT in this deploy. No kill-switch/env changes.

## 2026-06-24

- **~22:14 UTC (16:14 MT)** — `deb96e5` M-RS1 per-TF warmup + `dc053b3` M-RS2 Phase 1 bar-cache supply
  - Service(s) redeployed: api / Worker / frontend
  - Observed cache gap: (deploy window — expect ~1-2 min Worker restart gap)
  - Notes: New behavior behind kill-switches. RORT_RIGHTSIZE_WARMUP already set
    ON → M-RS1 becomes active with this deploy (recompute warmup right-sized).
    BAR_CACHE_ENABLED off (read path inert). New bar-cache maintain cron runs
    only if BAR_CACHE_MAINTAIN_ENABLED=true (set on Worker) → it will start
    keeping the configured TSLA 1s/1min targets current. New admin page at
    /admin/bar-cache. bar_cache_config table already migrated.

## 2026-05-12

- **~22:41 UTC (16:41 MT)** — `c2f7713` Fix algoMatches stale closure: deps missed algoTrades
  - Service(s) redeployed: frontend
  - Notes: Phase A had repointed `algoAll = algoTrades` inside the useMemo body but missed updating the deps array (still listed `[recentAlerts, fwdTrades, btTrades, ...]`). React never recomputed → algoMatches stayed empty → Chart & Trades Δ columns rendered `--` and Lab tab Price Divergence panel showed "No matched pairs." One-line dep array fix. Single useMemo at line 1927.

- **~21:44 UTC (15:44 MT)** — `27802ba` Divergence quick fixes: 24h default + clearer lane-update messages
  - Service(s) redeployed: frontend
  - Notes: Default Divergence tab window 48h → 24h per Kevin's request. Update New Data status messages now format per-lane outcomes (✓ appended +N / 0 new trades / skipped / error) so it's obvious whether a lane ran or didn't.

- **~20:46 UTC (14:46 MT)** — `66e1cf8` Divergence tab: date-window filter + admin page legend
  - Service(s) redeployed: api + frontend
  - Notes: Added `start`/`end` query params to `/divergence-data` (default 48h at the time). Date pickers + Last 24h/48h/7d/30d quick buttons on the tab. Admin page got an explicit legend explaining eRC/xCL drift columns.

- **~20:21 UTC (14:21 MT)** — `4b57710` Divergence reliability fixes: memory bounds, sort desc, theme styling
  - Service(s) redeployed: api + frontend
  - Notes: API OOM mitigation — `/divergence-data` + `/admin/divergence-summary` cap rows per lane (default 2000 / 1000), explicit `gc.collect()` between strategies in admin loop. Divergence rows sort desc (most recent first). Admin page styling fixed to use proper theme tokens.

- **~22:30 UTC (16:30 MT)** — `67675fb` Phase B: BT-APPEND true-append optimization
  - Service(s) redeployed: api
  - Notes: `append_new_backtest_trades_for_strategy` was doing a full DELETE+INSERT-all on every refresh. Now uses MAX(entry_fill_ts) query for anchor, per-row insert_trade for new rows, lazy KPI recompute. Expected 6-10× speedup on Update New Data: ~40min/batch → ~5min. New `db.get_max_entry_ts_admin` helper.

- **~22:00 UTC (16:00 MT)** — `976a041` Phase A: Chart & Trades wiring + admin divergence + Hi-Fi data-preservation fix
  - Service(s) redeployed: api + frontend
  - Notes: Critical Hi-Fi data-wipe fix (was overwriting trade `data` JSONB with `{hifi_resolved: true}` only on every refresh, destroying `bars_held`/`hold_time_seconds`/`pnl`/`win`/`behavior` etc.). New `/api/strategies/{id}/algo-trades` endpoint reads `cache_%` from trades table; Chart & Trades "Algo History" and "Price Divergence" modules now point at real algo lane data instead of mislabeled backtest. New admin page `/admin/divergence` for cross-strategy 3-way divergence summaries.

## 2026-05-11

- **~23:30 UTC (17:30 MT)** — `6b4fc53` Phase 41 fix: algo lane must scope DELETE + tag inserts with cache_<model>
  - Service(s) redeployed: api (algo-lane writers in forward_test_service)
  - Observed: evening Update All Data batch completed successfully across remaining ~16 strategies. Final trades-table distribution: 34,849 `backtest_rest_hifi`, 11,286 `cache_cache_locked`, 478 legacy `cache_locked`, 1,695 NULL orphans.
  - Notes: `recompute_and_persist_algo_trades` was wiping backtest rows the backtest lane had just written in the same bulk job (unfiltered DELETE) and inserting untagged rows. Same fix applied to `append_new_trades_for_strategy` (cron + forward-test path). Tag is `cache_<algo_model>` which produces the cosmetic-ugly `cache_cache_locked` since `algo_model='cache_locked'`; functionally fine via `cache_%` LIKE filter, cleanup queued.

- **~22:30 UTC (16:30 MT)** — `78d5597` Worker: throttle monitor_status writes to honor HEARTBEAT_INTERVAL
  - Service(s) redeployed: worker
  - Observed: monitor_status writes dropped from ~1800/hr to ~120/hr per active user. One contributor to Supabase load identified during today's investigation.
  - Notes: `db_write_status` was being called from `_periodic_tasks_loop` every PICKLE_WRITE_INTERVAL (~2s); the `HEARTBEAT_INTERVAL=30s` constant was never enforced. Now writes pass through immediately on state changes (running/connected flip) and otherwise honor the 30s throttle.

- **~22:00 UTC (16:00 MT)** — `d058653` Atomic-ish writer: retry transient errors + chunked INSERT
  - Service(s) redeployed: api + worker
  - Observed: sid 154 had failed earlier with Supabase 522 between DELETE and INSERT (data loss). This fix wraps `db.replace_trades_admin` + `db.insert_trade_admin` in `_execute_with_retry` (Cloudflare 5xx, "JSON could not be generated", connection/timeout — retried with 2s/4s/8s backoff; unique/FK violations skip retry). Bulk INSERT chunked at 250 rows. DELETE+INSERT idempotent on full retry.
  - Notes: addresses the recurring Supabase 522 outages from earlier in the day. Subsequent Update All Data run on sid 154 succeeded in 1,215s (slow due to Supabase still recovering, but completed cleanly via the retry path).

- **~18:30 UTC (12:30 MT)** — `8b74ddd` + `5fbfe0d` Phase 41 — backtest trades → trades table migration
  - Service(s) redeployed: api (writer + reader changes), worker (trades_store filter wiring)
  - Required manual steps: **Run `src/migrations/phase41_backtest_trades_relax_unique.sql` in Supabase SQL Editor BEFORE the backfill script**. Then run `python -m _backfill_stored_trades_to_table --apply` from `src/` to migrate existing stored_trades JSONB content into trades table with `data_source='backtest_<model>'`.
  - Observed cache gap: TBD
  - Notes: Completes the storage unification started in Phase 40 (2026-04-24). Backtest data now lives in the trades table alongside algo data, distinguished by `data_source` LIKE pattern (`backtest_%` vs `cache_%`). This unlocks REAL REST↔CACHE divergence in the Divergence tab — previously both lanes silently read from the same trades-table rows, producing fake "perfect alignment." Unique constraint relaxed to include `data_source` so REST + CACHE rows at identical timestamps coexist. Existing NULL data_source rows tagged as 'cache_locked' (preserves cron-written history). Backup: `dev-backup-pre-backtest-trades-migration-2026-05-11`.

## 2026-05-08

- **18:45 UTC (12:45 MT)** — `e7bf21a` + `0073dbb` + `0c208de` M8.7 Builder UIs + admin rename + REST-vs-CACHE design doc
  - Service(s) redeployed: api (mass_builder.py threads model kwargs through build_strategy_config), frontend (Strategy Builder + Mass Builder gain 3-dropdown picker; admin page + sidebar relabeled)
  - Observed cache gap: SUPABASE OUTAGE in progress (US-East-C network) — Railway/Vercel redeploys may queue; verification deferred
  - Notes: Pure frontend + safe backend additions during the Supabase outage. Strategy Builder + Mass Builder now expose `backtest_model` / `algo_model` / `live_model` selectors at creation time. Builder UIs are isolated — won't affect live engine. mass_builder.py:build_strategy_config now accepts model kwargs (None → GET-enrichment default). Admin "Backtest Models" page relabeled "Backtest / Algo Models" reflecting the shared registry. Design doc for v2 dual-storage divergence committed at `docs/Design_REST_vs_CACHE_Divergence_v2.md` — recommendation locks in Option B (JSONB map keyed by model). Backup: `dev-backup-pre-builder-uis-2026-05-08`.

- **16:30 UTC (10:30 MT)** — `d867e63` + `5fddf86` M8.7 lane×mode matrix complete + button uniformity
  - Service(s) redeployed: api (2 new helpers + JWT-safe /update endpoint), recompute_jobs worker (bulk fans to both lanes), frontend (Strategy Detail header buttons unified)
  - Observed cache gap: TBD
  - Notes: All 4 lane×mode combinations now wired in /update endpoint. Adds `append_new_backtest_trades_for_strategy` (forward append → stored_trades JSONB under backtest_model) and `recompute_and_persist_algo_trades` (full recompute → trades table under algo_model). JWT expiration on long backtest runs fixed via `set_admin_user_context` wrapper. Bulk page job-worker also fans out to both lanes per Option A. Strategy Detail header now has Update New Data + Update All Data instead of Refresh. Live model labeling on alerts verified working — 20/20 alerts since 2026-05-07 22:00 UTC carry `live_model='ws_agg_locked'`. Backup: `dev-backup-pre-lane-mode-matrix-2026-05-08`.

## 2026-05-07

- **23:10 UTC (17:10 MT)** — `3bf6911` + `a6d0bb8` M8.7 algo_model split + live_model labeling + Divergence buttons
  - Service(s) redeployed: api (algo_model field, /update endpoint, divergence response), Worker (DBAlertDispatcher + AlertDispatcher stamp live_model on alert fire), frontend (Update buttons + lane badges + Live model column)
  - Observed cache gap: TBD
  - Notes: TWO migrations need to run on Supabase: `alerts_live_model_column.sql` (adds live_model TEXT column to alerts) and previously-applied `algo_history_cron_cycles.sql`. Bulk migration `_bulk_set_algo_model.py --apply` already ran — all 20 strategies now backtest_model=rest_hifi + algo_model=cache_locked. Sid 152 flipped from cache_locked-backtest. Live alerts going forward carry live_model in payload; legacy alerts render "unknown" (no backfill). Backup: `dev-backup-pre-algo-model-2026-05-07-evening`. Two of four lane×mode update combinations deferred (forward backtest append + full algo recompute).

- **21:25 UTC (15:25 MT)** — `cd58026` M8.7 Divergence tab: 3-lane comparison on Strategy Detail
  - Service(s) redeployed: api (new endpoint + service), frontend (new tab + hook + types)
  - Observed cache gap: TBD
  - Notes: Adds GET /api/strategies/{id}/divergence-data + new "Divergence" tab on Strategy Detail. Three lanes today: Backtest (stored_trades JSONB) / Algo (trades table from cron) / Live (alerts). KPIs + drift stats (median/p95/max) per lane pair; color-coded ≤2s green / ≤30s yellow / >30s red. Real REST-vs-CACHE comparison deferred to v2 (needs dual-storage backend). Backup: `dev-backup-pre-divergence-tab-2026-05-07`.

- **19:47 UTC (13:47 MT)** — `1dbf334` + `57ec22a` + `2245cc7` M8.7 cron stats panel + pack manifest cleanup + EOD docs
  - Service(s) redeployed: Worker (new DB writes), api (new endpoint + DB helpers), frontend (Jobs page CronStatsPanel)
  - Observed cache gap: TBD — pushed at 19:47 UTC
  - Notes: Three commits in one push. (1) `2245cc7` adds `trigger_levels_phase2` markers to 7 cross-style packs (cleanup; no engine path change). (2) `57ec22a` is the cron-cycle stats panel — Worker now writes one row per user per cycle to `algo_history_cron_cycles` (best-effort, swallows failures), prunes to last 200 rows every 10th cycle. New API endpoint `GET /api/jobs/cron-stats/algo-history`. Frontend: CronStatsPanel above Jobs list with Fresh/Normal/Stale buckets, last-5-cycle insertion totals, Starvation flag when oldest stamp >3x cycle interval. Migration `algo_history_cron_cycles.sql` already applied on Supabase by Kevin. (3) `1dbf334` is docs/EOD writeup + Roadmap milestones.

- **18:01 UTC (12:01 MT)** — `508cb89` M8.7 AI pack-creation guardrail: trigger_levels enforcement
  - Service(s) redeployed: api (pack_spec/pack_registry/pack_builder updates)
  - Observed cache gap: none expected (additive validation)
  - Notes: Three-layer guard — `pack_spec.audit_trigger_levels()` regex, install-time warnings via `pack_registry.scan_and_load_all`, AI builder treats audit warnings as errors in `validate_parsed_response`. New manifest schema field `trigger_levels_phase2` for intentional non-static markers.

- **17:51 UTC (11:51 MT)** — `169df09` M8.7 Hi-Fi Phase 1: signal-exit refinement via user-pack trigger_levels
  - Service(s) redeployed: api (Hi-Fi Pass 2 endpoints)
  - Observed cache gap: none expected (refinement is endpoint-driven, not cron-fired automatically yet)
  - Notes: New `_walk_1s_for_level_cross` walker + `pack_registry.get_trigger_level_spec()` resolver + `bar_df` plumbed into `_hifi_resolve_trades`. Covers `eppv3`/`eppv4`/`utv4` packs (declare static-level cross semantics). Phase 2 (indicator-vs-indicator / value-vs-threshold / dynamic) deferred — design seed includes new exec_type X discussion.

## 2026-05-04

## 2026-05-05

- **16:05 UTC (10:05 MT)** — env flag `WS_AGG_SHADOW_ENABLED=true` set on Worker (no commit)
  - Service(s) redeployed: Worker (auto-triggered by env-var change)
  - Observed cache gap: ~1-2 min during Worker restart
  - Notes: Activates the WsAggMinuteBuilder shipped at `ace1fe7`. After warmup, expect `source='ws_agg'` 1Min rows to appear in live_bars for SPY (10Sec subs already trigger A.* subscription) and TSLA. Backup branch `dev-backup-pre-wsagg-flag-on-2026-05-05` pushed pre-flip.

- **21:01 UTC (15:01 MT)** — `ee2c4c5` empty commit to recover from stuck/failed builds at 19:42-19:44
  - Service(s) redeployed: Worker (clean rebuild), frontend (auto-recovered)
  - Observed cache gap: ~3 min during Worker restart
  - Notes: 3 rapid pushes earlier (Phase C wire + default flip + bulk script) caused Railway to queue 3 conflicting builds — 2 stuck BUILDING + 1 FAILED. Worker kept serving the 12:52 deploy (Phase C code, no default flip) until this empty-commit kicked a clean rebuild. Post-recovery verification: all 27 monitored strategies have explicit `config.live_model='ws_agg_locked'`, 12 Polygon channels subscribed (6 AM + 6 A), Phase C dispatch confirmed working for AAPL/AMD/SPY/TSLA. META/TSLL ws_agg=0 due to A.* events only firing on actual trade activity (low post-market volume) — expected behavior, will normalize during RTH. 1Min alerts silent post-market because most are session=RTH and we're past 16:00 ET — also expected.

- **19:45 UTC (13:45 MT)** — `f971eeb` + `b1ad5c0` M8.7 Phase C: live_model default flipped to ws_agg_locked + bulk DB migration
  - Service(s) redeployed: Worker (auto-rebuilds on default flip code change)
  - Observed cache gap: ~1-2 min during Worker restart
  - Notes: Default live_model flipped from ws_with_corrections → ws_agg_locked. Bulk update via _bulk_set_live_model.py set explicit `config.live_model='ws_agg_locked'` on all 39 strategies. End state: every monitor reports ws_agg_locked, A.* subscribes universally on next reconnect (4 → ~9 symbols, 2.25× per-second forming-bar load — within budget). Backup: `dev-backup-pre-phase-c-2026-05-05` + snapshot `/tmp/strategies_live_model_snapshot_20260505T194258Z.json`. Revert path: re-run `_bulk_set_live_model.py NULL` then revert default in code (~3 min). Supabase had a 522/525 outage during prep window; user restarted to recover.

- **18:55 UTC (12:55 MT)** — `74ff56c` M8.7 Phase C: live engine dispatch + universal A.* subscription
  - Service(s) redeployed: Worker
  - Observed cache gap: ~1-2 min during Worker restart (RTH session active)
  - Notes: Wires live_model into StrategyMonitor + bar-close gate + ws_agg dispatch path. A.* subscription gate gains has_ws_agg condition (selecting ws_agg_locked on a strategy auto-subscribes A.<symbol>). Flipped ws_agg_locked.available=True so ModelsCard offers it. Default UNCHANGED at ws_with_corrections to avoid re-triggering M8.5 forming-bar latency regression. Backup branch `dev-backup-pre-phase-c-2026-05-05` pushed pre-change.

- **16:30 UTC (10:30 MT)** — `2ee8a24` M8.7 Phase A+B: strategy model registry + admin pages
  - Service(s) redeployed: api (new strategy_models_admin router), frontend (two new admin pages + sidebar entries)
  - Observed cache gap: N/A (additive, no engine code changes)
  - Notes: Refreshes strategy_models.py with ws_agg_locked / ws_agg_with_rest_backfill (live) and cache_locked / cache_corrected (backtest), all `available=False` until Phase C/D/E ship. Removes 3 unused M3 placeholder backtest IDs. New admin routes /admin/live-models and /admin/backtest-models surface usage counts + per-model strategy lists. Backup branch `dev-backup-pre-strategy-models-2026-05-05` pushed pre-change. Plan archived at `~/.claude/plans/synchronous-tickling-yeti.md`.

- **15:50 UTC (09:50 MT)** — `ace1fe7` WsAggMinuteBuilder shadow (disabled by default)
  - Service(s) redeployed: Worker (no-op until `WS_AGG_SHADOW_ENABLED=true`)
  - Observed cache gap: N/A (additive code path, env flag off)
  - Notes: Lands the Mode-2 candidate aggregator + validator. To activate: `WS_AGG_SHADOW_ENABLED=true` on Railway Worker, restart, wait ~30 min for paired bars, run `src/_validate_ws_agg.py` from inside `src/`. Backup branch `dev-backup-pre-wsagg-2026-05-05` pushed pre-change. Phase 1 scope: only writes for symbols where A.* is already subscribed (SPY, TSLA today).

## 2026-05-04

- **23:35 UTC (17:35 MT)** — `e30bfc6` M8.7 admin/data-health: custom date range filter
  - Service(s) redeployed: api (data_health router accepts start/end), frontend (V1 mode toggle + datetime picker)
  - Observed cache gap: N/A (read-only diagnostic)
  - Notes: Rolling mode (1h/4h/RTH/24h) is default; Custom mode reveals datetime-local inputs in user's chartPrefs.timezone with sec precision. Useful for "show me coverage since the last deploy without rolling-window noise." TV CSV stability test result also recorded — TV revises closed bars within ~5 min, confirming `live_model='latest'` should remain default.

- **23:30 UTC (17:30 MT)** — `6291a9a` M8.7 admin/data-health: per-(symbol, tf) cache coverage dashboard
  - Service(s) redeployed: api (new router), frontend (new page + sidebar entry)
  - Observed cache gap: N/A (read-only diagnostic page)
  - Notes: Backs the "is data being collected" question. Surfaces AM-stream loss pattern (AAPL/AMD/SPY 1Min ~35-40% coverage vs 10Sec 97%) plus subscribed-but-empty entries (GME, INTC, NVDA, TSLA/1Min). Backup branch `dev-backup-pre-data-health-2026-05-04` pushed.

- **23:15 UTC (17:15 MT)** — `1cca741` M8.7 M5: Custom picker uses user's tz + Apply forces refit
  - Service(s) redeployed: frontend (next.js)
  - Observed cache gap: N/A (frontend-only)
  - Notes: Custom datetime inputs now format/parse using `chartPrefs.timezone` (matches chart axis 1:1) with DST-aware two-pass refinement. Apply button now triggers chart refit by remounting via React `key` keyed on `${windowStart}-${windowEnd}` — fixes the silent-no-update bug where setData preserved the previous visible range across window commits.

- **22:50 UTC (16:50 MT)** — `2d6ea98` M8.7 M5: Custom timestamp picker + bar-count diagnostic
  - Service(s) redeployed: frontend (next.js)
  - Observed cache gap: N/A (frontend-only)
  - Notes: Adds explicit start/end datetime-local inputs (sec precision) so trader can target a specific trade window for forensic diagnosis. Plus robust empty/inverted-intersection handling and per-lens bar counts in header — strat 149 first-write empty case now shows "Algo X / Alert 0 bars" with diagnostic message instead of blank card. Backup branch `dev-backup-pre-timestamp-picker-2026-05-04` pushed pre-change.

- **22:25 UTC (16:25 MT)** — `0c368e8` M8.7 M5: lens-extent intersection + trigger_prefix relevance match
  - Service(s) redeployed: frontend (next.js), api (strategies router)
  - Observed cache gap: N/A (frontend + API logic only)
  - Notes: Two fixes from v2 smoke test. (a) LabReplayPanel uses INTERSECTION of both lenses' candle extents — Algo and Alert now share start AND end candles, eliminating the REST-trailing-WS visual mismatch. (b) `_build_chart_response_from_df` relevance check gains a third path: match via template's `trigger_prefix`. Fixes EPP v4 (and any user pack where group.id ≠ trigger_prefix) — affects both Chart & Trades and Lab tab.

- **22:00 UTC (16:00 MT)** — `14e211a` M8.7 M5 v2: unified Lab Replay panel
  - Service(s) redeployed: frontend (next.js)
  - Observed cache gap: N/A (frontend-only)
  - Notes: Architectural pivot after smoke-test feedback. SyncedChartPane gets a `currentTime` prop for scrub mode (additive); new LabReplayPanel wraps two SyncedChartPanes (Algo REST + Alert cache) sharing one scrub head + window picker (Last 1h / 4h / Today / All). Replaces V1 ChartReplayCard's parallel renderer — indicators, oscillators, heatmap and markers all render identically on both lenses now. Net -290 lines on StrategyDetailPage.

- **19:40 UTC (13:40 MT)** — `dc91ba9` M8.7 M5: Replay marker parity (algo + and alert × price-level crosses)
  - Service(s) redeployed: frontend (next.js)
  - Observed cache gap: N/A (frontend-only)
  - Notes: Round 2 of M5 smoke-test feedback. Replay now mirrors Chart & Trades' price-level cross markers (4 invisible Line series with shape='cross' for algo, 'xcross' for alert; filtered by replay scrub time). Indicator overlays still empty for strats whose user packs don't expose chart columns (e.g. ema_pp_v4) — separate backend issue.

- **18:20 UTC (12:20 MT)** — `8232289` M8.7 M5 fix: Replay parity (candleCount slice + trade markers + rightOffset)
  - Service(s) redeployed: frontend (next.js)
  - Observed cache gap: N/A (frontend-only)
  - Notes: Monday smoke test found V1 Replay rendering candles only. Fix slices to candleCount, builds entry/exit markers, forwards rightOffset, and keys chart id on overlay count. Plan + memory updated. Backup branch `dev-backup-pre-replay-fix-2026-05-04` pushed.

- **15:13 UTC (09:13 MT)** — `53cac54` M8.7 M4 fix: carry indicator_snapshot through DBAlertDispatcher
  - Service(s) redeployed: Worker
  - Observed cache gap: TBD (RTH push — expect ~1-2 min reconnect window)
  - Notes: Monday RTH validation found 0/15 recent alerts had `data.indicator_snapshot`. Saturday's M4 only patched `AlertDispatcher.dispatch` in ralph_engine.py, but worker.py overrides `engine.dispatcher` with `DBAlertDispatcher`. Mirror added — same 3-line pattern. Validate post-restart by checking `data->indicator_snapshot` on next RTH alerts.

## 2026-05-02

- **18:30 UTC (12:30 MT)** — `5515401` Archive M8.7 Saturday plan + update Roadmap_To_Scale with M1-M6 status
  - Service(s) redeployed: all (docs-only push)
  - Observed cache gap: ~1-2 min during Worker restart (markets closed)
  - Notes: pure documentation. Plan archived in repo. Roadmap reflects shipped vs remaining work.

- **17:57 UTC (11:57 MT)** — `064f9a3` M8.7 M4+M5+M6: alert snapshots, Lab tab replay, engine-state capture
  - Service(s) redeployed: Worker (M4 snapshot, M6 writer hook), api (no functional change), frontend (M4 tooltip, M5 replay)
  - Observed cache gap: ~1-2 min during Worker restart (markets closed)
  - Notes: M4-M6 of weekend plan. Required manual steps: (1) apply src/migrations/bar_engine_states_table.sql in Supabase, (2) set BAR_ENGINE_STATE_WRITE_ENABLED=true on Worker.

- **16:49 UTC (10:49 MT)** — `8b40eff` M8.7 M3: backtest_model + live_model placeholder (schema + UI)
  - Service(s) redeployed: api (new endpoint + default fill), frontend (Models card + badges), Worker (no functional change)
  - Observed cache gap: ~1-2 min during Worker restart
  - Notes: M3 of weekend plan. Models abstraction recorded on each strategy; engine dispatch comes later. UI: ModelsCard on Config tab, read-only badges in page header.

- **16:44 UTC (10:44 MT)** — `7bd71e7` M8.7 Phase 2: Lab tab Alert Lens uses cache-derived indicators + heatmap
  - Service(s) redeployed: api (new endpoint + refactored helper), frontend (new hook + Lab tab wiring), Worker (no functional change)
  - Observed cache gap: ~1-2 min during Worker container restart (markets closed)
  - Notes: M2 of weekend plan. Closes the Phase 1 caveat — Alert Lens indicators/heatmap now computed from live_bars cache. New endpoint /chart-data-cache; refactored _build_chart_response_from_df shared with /chart-data.

- **16:36 UTC (10:36 MT)** — `a15461f` M8.7 fix: handle Polygon WS rebroadcasts as corrections, not duplicates
  - Service(s) redeployed: Worker (engine code change), api/frontend rebuilt
  - Observed cache gap: ~1-2 min during Worker container restart (markets closed → no live trading impact)
  - Notes: M1 of weekend plan. Fixes the duplicate-bar bug discovered Friday EOD. accept_bar/accept_second_bar now detect rebroadcasts and replace history rows instead of appending. IncrementalIndicatorEngine gets new recompute_from_history method. Worker restart rebuilds in-memory state from REST warmup (clean). Live validation deferred to Monday RTH.

## 2026-05-01

- **20:13 UTC (14:13 MT)** — `671dd6b` EOD checkpoint: weekend plan, deploy log, M8.7 findings + TV-cache compare data
  - Service(s) redeployed: all (docs-only push but Railway watchPatterns=[])
  - Observed cache gap: ~1 min during Worker container restart
  - Notes: pure documentation commit. No code/engine changes. Captures Friday's findings (duplicate-bar bug, TV≈REST, TF drift scaling) and Saturday's plan.

- **19:41 UTC (13:41 MT)** — `5ba2dd7` Lab tab: Phase 1 — side-by-side Algo Lens / Alert Lens
  - Service(s) redeployed: frontend (Worker rebuilt but no functional change)
  - Observed cache gap: ~1 min during Worker restart
  - Notes: replaces single-chart toggle with always-visible side-by-side layout. Right side has its own First/Latest sub-toggle. Phase 1 ships candle differences only — indicators/heatmap on right are still REST-derived (Phase 2 will fix).

- **19:23 UTC (13:23 MT)** — `106f1ca` Lab tab: data-source toggle for live-WS chart view (M8.7)
  - Service(s) redeployed: api (new endpoint), frontend (new hook + UI). Worker rebuilt but no code change there.
  - Observed cache gap: ~1 min during Worker restart. None expected from api/frontend redeploy.
  - Notes: ships the read-only side of M8.7. Backend `/cache-bars` endpoint reads from live_bars; frontend toggle on Lab tab swaps `labChartTabData.bars` between REST / WS-latest / WS-first. Indicators/heatmap stay REST-derived (full read path = M8.7d, not in scope).

- **19:06 UTC (13:06 MT)** — `ef28f0b` Strategy Detail: add 'Chart & Trades (Lab)' tab for divergence visualization
  - Service(s) redeployed: frontend (api/Worker/streamlit also rebuilt — Railway redeploys all on dev push, but no functional change to those)
  - Observed cache gap: minimal — frontend redeploy doesn't affect Worker bar recording. Worker container also rebuilt; ~1-2 min restart gap if affected.
  - Notes: pure frontend addition. New tab parallel to existing Chart & Trades. Reuses existing chart data and trade-to-alert matching logic. Adds a Price Divergence panel surfacing algo vs alert price gap per matched pair.

- **18:30 UTC (12:30 MT)** — `21cd636` M8.7 hotfix #2: don't overwrite 1Min canonical bars with stale partials
  - Service(s) redeployed: Worker (deploy SUCCESS); api, frontend, streamlit too — every dev push redeploys all
  - Observed cache gap: 1-2 min around 18:30 UTC during Worker container restart
  - Notes: yesterday's hotfix added a flush_stale_bars write hook that fired correctly for sub-minute primaries but also fired incorrectly for 60s+ builders, overwriting canonical AM bars with chart-visual partial data. This fix gates the flush write to `tf_seconds < 60`. Cleanup of 346 corrupted 1Min rows (volume <50% of first_volume) executed inline post-deploy — restored from `first_*` columns. Going forward any `first_close ≠ close` in the 1Min cache reflects a genuine Polygon WS rebroadcast correction, not the bug.

- **15:21 UTC (09:21 MT)** — `760b64c` trigger Worker redeploy (empty commit)
  - Service(s) redeployed: Worker (api unaffected)
  - Observed cache gap: `15:23:00` 1Min bar missing across AAPL/AMD/META/SPY/TSLL — confirmed by `_validate_live_bars_cache.py`
  - Notes: needed because the prior `2dd409b` hotfix deploy was stuck in BUILDING with `deploymentStopped: true` (Railway state inconsistency). Empty commit forced a fresh build that succeeded ~15:24 UTC.

- **15:11 UTC (09:11 MT)** — `2dd409b` M8.7 hotfix: add live_bars write hook to flush_stale_bars path
  - Service(s) redeployed: Worker (build cancelled — never went live)
  - Observed cache gap: none (deploy never completed, old code kept running)
  - Notes: build entered `BUILDING` then got marked `deploymentStopped` without progressing. Required the empty-commit retrigger above.

## 2026-04-30

- **18:46 MT (00:46 UTC, 2026-05-01)** — `e6d946f` M8.7: live_bars cache write path (fire-and-forget, flag-gated)
  - Service(s) redeployed: Worker, api
  - Observed cache gap: n/a (this was the first deploy that introduced live_bars; nothing to gap before it)
  - Notes: initial M8.7 ship. `LIVE_BAR_CACHE_WRITE_ENABLED=true` set on Worker via `railway variables --skip-deploys` immediately before. Deploy succeeded cleanly in ~50s.

---

## How to use this log

When investigating a cache gap, alert miss, or WebSocket reconnect:

1. Note the affected timestamp window from the data.
2. Check this log for any entry within ±5 min of that window.
3. If a deploy lines up, the gap is expected and you can move on.
4. If no deploy lines up, dig deeper — it's a real issue.

When you (Claude) push code that triggers a Worker / api / frontend
redeploy, append a new entry at the **top** of the most recent date
section before reporting back to Kevin. Pull the commit SHA and time
from `git log` after pushing. Notes about observed gaps can be added
later once validated.
