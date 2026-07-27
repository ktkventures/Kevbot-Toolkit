# Team Board — MIGRATED to the app (frozen snapshot)

**⚰️ TOMBSTONE 2026-07-22:** the live team board now lives at **`/admin/tasks`** in the
app UI (dev_tasks table; PR #71). All assignments, status updates, and new tasks happen
THERE — vision items are `V#`-titled parents tagged `vision`; fixes discovered mid-work
get created as subtasks (`parent_id`) with `origin='discovered'`. The SessionStart hook
serves role queues from the API. Do NOT update the tables below — they are a frozen
07-22 snapshot kept only as the hook's offline fallback. Rules live in charter §7.

---

## V1 — Trading at scale: divergence → 90% paired (lane: E)
Master checklist: `docs/_active/Roadmap_Trading_At_Scale.md`

| # | Task | Assignee | Status | Notes |
|---|------|----------|--------|-------|
| V1.1 | ~~Kevin decision: container-memory fix~~ | Kevin | MOOT 21:42Z | Premise retracted (true cause = stale-tarball var-set rebuilds resurrecting July-20 code); V1.2 re-attempt on current code: FLEET STREAMS 23/23 in 10 min. No container spend needed. Residual (non-blocking): bound the KPI-HiFi whole-history load-once (345 ×8-pool crash class) — proposed as new V1.8 |
| V1.2 | Re-attempt fleet SIDS flip on CURRENT code | E2 | ✅ DONE 21:42Z | **FLEET STREAMING 23/23** @POLL_S=60 (bootstrap 6→19→23 in 10 min; fingerprint f4fd3ab24a5e; tracer aboard). HARDENED SOP (permanent): every shadow-worker var-set → immediately railway up + fingerprint — var-set rebuilds resurrect the stale pinned tarball |
| V1.8 | Bound KPI-HiFi whole-history load-once (NVDA 7.4M rows/load; 345 ×8-pool crash class; batch pool can return to 8 after) | E2 | TODO (non-blocking) | Proposed by E2 21:45Z from today's forensics |
| V1.3 | M-RS5a canary watch | E1 | DONE 07-22 | Canary validated (byte-identical, 2h healthy, POLL_S=60, kill switch ready); E1 retired 21:17Z clean. ⚠ Canary flag remains DELIBERATELY ARMED on shadow-worker (RESIDENT_FRAME=1, sids 263/267/269/271) — now E2's watch |
| V1.4 | Verify tonight's nightly from-cold diff = 0 | E2 | TODO | E1's leftover, transferred on retirement |
| V1.5 | POLL_S=5 restore | E2 | SCHEDULED 07-23 late-AM | Unblocked by V1.2; one-knob-per-day: fleet soaks overnight @60s (nightly + tomorrow open), then 5s if clean. SOP: var-set + immediate railway up |
| V1.6 | Sub-min canonical (339) live-vs-NEW-settled validation + gate-4 XTF_BLOCK_DIAG | E | ACTIVE | Gate-4 GREEN 14:19:33Z per E2's sync; 339 pairing continues as live alerts accumulate |
| V1.7 | Provisional backtest tail (design → build) | E | TODO | `Design_Provisional_Backtest_Tail.md`; long-term target per Kevin 07-22; after fleet promotion |

## V2 — Frontend reporting (lane: F)

| # | Task | Assignee | Status | Notes |
|---|------|----------|--------|-------|
| V2.1 | Spawn F session in `../Kevbot-frontend` worktree | Kevin | DONE 07-22 | F live on `feat/tasks-team-board` @0a31a9b |
| V2.4 | Tasks-page team-board implementation | F | ACTIVE — PR READY 07-22 | **PR #71 ready + release brief handed to Kevin** (spec Phase 1 complete: additive migration APPLIED to Supabase, API nesting/assignee filter, grouped UI — all acceptance criteria verified vs live API); awaiting R merge, then M does the data migration (spec §4) |
| V2.2 | Health-overview page UI improvements (feeds bug-hunt observability) | F | TODO | After V2.4, new branch — audit what the hunt loop needs surfaced |
| V2.3 | Portfolio pages overhaul | F | TODO | |

## V3 — User packs (lane: P) — HOLD until V1.1/V1.2 settle + board green

| # | Task | Assignee | Status | Notes |
|---|------|----------|--------|-------|
| V3.1 | S/R pack accuracy audit + fixes | P | HOLD | Pack edits shift paired-% baselines — timing coordinated with E |
| V3.2 | New user packs | P | HOLD | |
| V3.3 | Pack-builder AI refresh (in-app creation flow up to date) | P | HOLD | |

## V4 — Team infrastructure (lane: M)

| # | Task | Assignee | Status | Notes |
|---|------|----------|--------|-------|
| V4.1 | Charter + naming + skills (/role-handoff, /release-brief) | M | DONE 07-22 | This board = interim home base |
| V4.2 | Spec: Tasks page as team board (roles, parent/subtask, status, origin) | M | DONE 07-22 | `Spec_Tasks_Team_Board.md` — existing dev_tasks table/API/UI covers ~70%; additive migration + grouped UI |
| V4.3 | Spec: admin-UI roadmap page (visual big picture for Kevin) | M | TODO | Phase 2 of `Spec_Tasks_Team_Board.md`; detail after V2.4 ships |
| V4.4 | `docs/_active/` hygiene pass (archive stale, index the rest) | M | TODO | ~60 files and growing |
| V4.5 | Commit charter/board/skills so worktree sessions inherit them | E | DONE 07-22 | dev `0a31a9b` (docs+skills only), all 4 services rebuilt green; F worktree fast-forwarded by M — F inherits skills |
| V4.6 | Track `.claude/skills/session-handoff/` in git (role-handoff builds on it; missing from worktrees) | E | TODO | Fold into E's next routine dev push — one `git add` |
| V4.7 | SessionStart hook: auto-inject the role's open board rows at session start/resume (markdown grep now; `/api/dev-tasks?assignee=` after V2.4) | M | DONE 07-22 | Script `.claude/hooks/team_board_context.py` (main checkout, absolute-path shared); wired via settings.local.json in main + both worktrees; role markers `.claude/role` (F, E2); main-checkout = whole-board digest grouped by role (one-window workflow). Takes effect on each session's NEXT start/resume. Post-V2.4: point script at the API |
