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

## 2026-07-25

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
