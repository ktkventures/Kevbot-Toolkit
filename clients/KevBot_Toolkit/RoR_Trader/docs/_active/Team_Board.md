# Team Board — assignments & big picture

**Home base for the multi-session team** (interim markdown version; target = the app's
Tasks page once specced by M and built by F — see charter §7).
Curated by **M**. Other roles: update status on YOUR lane's rows only, at session start
and on completion. New fixes/follow-ups discovered mid-work get parented UNDER the
vision item that spawned them — never as free-floating tasks. That rule is how the big
picture survives the rabbit holes.

Statuses: `TODO` · `ACTIVE` · `BLOCKED(<on what>)` · `HOLD` · `DONE <MM-DD>`
Operational detail lives in `STATUS.md` / lane plan docs — this board tracks WHO is on
WHAT and how it ladders up, not the play-by-play.

---

## V1 — Trading at scale: divergence → 90% paired (lane: E)
Master checklist: `docs/_active/Roadmap_Trading_At_Scale.md`

| # | Task | Assignee | Status | Notes |
|---|------|----------|--------|-------|
| V1.1 | Kevin decision: container-memory fix (resize vs bounded KPI-HiFi window vs fleet-minus-heavies) | Kevin | ⚠ PREMISE RETRACTED 21:3xZ | E2 A/B post-close: all 3 railway-up boots stream, all 5 var-set rebuilds dead ⇒ TRUE cause = Railway var-set rebuilds from STALE July-20 tarball (old code = the original freeze bug). 18:22Z fleet "failure" ran OLD code. Container-memory unproven — decision likely moot pending V1.2 re-attempt; bounding KPI-HiFi loads stays worthwhile (345 crash class) but non-blocking |
| V1.2 | Re-attempt fleet SIDS flip on CURRENT code (SOP: var-set → immediate railway up + fingerprint) | E2 | IN PROGRESS 21:32Z | Re-attempt launched (SIDS=all + railway up, fingerprint f4fd3ab24a5e, tracer aboard); watcher logs progress /3min, 2h bound; revert = SIDS var-set + railway up. Hardened SOP: EVERY shadow-worker var-set must be followed by railway up + fingerprint |
| V1.3 | M-RS5a canary watch | E1 | DONE 07-22 | Canary validated (byte-identical, 2h healthy, POLL_S=60, kill switch ready); E1 retired 21:17Z clean. ⚠ Canary flag remains DELIBERATELY ARMED on shadow-worker (RESIDENT_FRAME=1, sids 263/267/269/271) — now E2's watch |
| V1.4 | Verify tonight's nightly from-cold diff = 0 | E2 | TODO | E1's leftover, transferred on retirement |
| V1.5 | POLL_S=5 restore | E2 | BLOCKED(V1.2) | |
| V1.6 | Sub-min canonical (339) live-vs-NEW-settled validation + gate-4 XTF_BLOCK_DIAG | E | ACTIVE | Gate-4 GREEN 14:19:33Z per E2's sync; 339 pairing continues as live alerts accumulate |
| V1.7 | Provisional backtest tail (design → build) | E | TODO | `Design_Provisional_Backtest_Tail.md`; long-term target per Kevin 07-22; after fleet promotion |

## V2 — Frontend reporting (lane: F)

| # | Task | Assignee | Status | Notes |
|---|------|----------|--------|-------|
| V2.1 | Spawn F session in `../Kevbot-frontend` worktree | Kevin | TODO | Worktree created 07-22 by M (branch `feat/health-overview-ui` @7783548); open `/home/kevin/projects/Kevbot-frontend` in VS Code, paste opener, name it `F — Frontend` |
| V2.2 | Health-overview page UI improvements (feeds bug-hunt observability) | F | TODO | First deliverable — audit what the hunt loop needs surfaced |
| V2.3 | Portfolio pages overhaul | F | TODO | |
| V2.4 | Tasks-page team-board implementation | F | BLOCKED(V4.2 spec) | |

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
| V4.2 | Spec: Tasks page as team board (roles, parent/subtask, status, origin) | M | TODO | → `Spec_Tasks_Team_Board.md`; F builds |
| V4.3 | Spec: admin-UI roadmap page (visual big picture for Kevin) | M | TODO | Pairs with V4.2 |
| V4.4 | `docs/_active/` hygiene pass (archive stale, index the rest) | M | TODO | ~60 files and growing |
| V4.5 | Commit charter/board/skills so worktree sessions inherit them | Kevin/E | TODO | Files currently uncommitted in main checkout |
