-- Agents Registry — split `R` into R (session conductor) + R-A (headless executor).
-- Board #229, step 2 (F) · 2026-07-30 · design: Kevin, 07-30.
--
-- DATA, NOT SCHEMA. No DDL: both statements are ordinary INSERT/UPDATE against
-- columns `agents` has had since agents_registry.sql (2026-07-23). Exactly the
-- shape board #222 used for M/M-A three hours earlier, deliberately — the point
-- of doing the second split this way is that it costs no new mechanism at all.
--
-- WHY — and it is NOT that R·auto is bad at its job
-- Measured: 27 runs, 24 `ok`, 17 trains shipped. The judgement is disciplined —
-- it aborts on red gates rather than improvising (#217, #194, #191), self-reports
-- its own deviations, watches deploys to settlement, and independently confirmed
-- a RED on #226 before trusting the fix. The problem is the OTHER side of that
-- discipline: every correct hand-back lands on M. On 07-30 alone M absorbed a
-- stale rail-5 reconcile (#217), a false-negative refusal override (#222), a
-- same-line rebase on Car 3, THREE rewrites of Wave 21's brief, a wave-number
-- collision and a Railway incident. None of that is orchestration; all of it is
-- release-lane technical work sitting in the coordinator's context.
--
-- So the lane splits by CONTEXT HORIZON, not by competence (Kevin, 07-30):
--   R   — the conductor. Reads the `Staged` queue and decides what ships
--         together and in what order; authors and ADAPTS briefs mid-flight;
--         reconciles red gates and conflicts; calls incidents; holds the
--         cross-train picture (what supersedes what, wave numbering). Retains
--         long-term memory, and is THE FALLBACK WHEN AUTOMATION FAILS.
--   R-A — the executor. One train, bounded context: merge → gate → deploy-watch
--         → log, per a brief. Runs gates twice, verifies before merging, and
--         ABORTS AND HANDS BACK rather than improvising.
-- M's remaining role in releases: mark a task `Staged`. Everything after that
-- is R's. That is the load this removes.
--
-- HOW IT FIXES IT — no new mechanism
-- `dispatcher.headless_agents()` selects `status=eq.headless`. Flipping `R` to
-- 'live-session' drops it out of that set, and `triage_todo()` already has the
-- correct behaviour for an assignee that is not a headless agent: skipped with
-- `is_stuck=False` — "waiting on R (not a headless agent)", quiet, no staleness
-- tripwire and no repeated board comment. `kevin` has proven that path since
-- 07-23 and `M` has sat on it since the #222 flip landed earlier today.
--
-- ORDERING (deliberate — read before applying)
--   Statement 1 (INSERT R-A) is ADDITIVE and safe to apply at any time: it adds
--   a lane nothing is assigned to yet. R keeps dispatching until statement 2.
--   Statement 2 (UPDATE R) is the BEHAVIOUR CHANGE. The moment it lands, every
--   open task assigned `R` stops dispatching — including any release train
--   mid-flight. Board #229 step 3 (M classifies R's open work as R or R-A)
--   exists precisely so that flip does not strand the release lane. Apply
--   statement 2 with the release train (step 4), AFTER the classification.
--   Per board #222's second lesson: DISARM before restatusing anything armed
--   while this is being applied — two races on 07-30 came from changing an
--   armed task's status, and R touches statuses constantly.
--
-- KNOWN FOLLOW-UP, deliberately NOT done here (M's call, not F's)
--   R's own `scope`/`name` prose still reads "Ephemeral gatekeeper: validate →
--   merge → deploy-watch → log → die. Nothing persistent" — which describes
--   R-A exactly (hence the inheritance below) and describes the CONDUCTOR not
--   at all. Rewriting it is a judgement call about the session's remit, so it
--   is raised on the task rather than smuggled into a flip whose only job is
--   `status`. The charter §1/§2 carries the correct division in the meantime.
--
-- Idempotent: re-running is a no-op.

-- 1. R-A inherits R's scope, boundaries, worktree, context docs and prompt
--    template BY SELECTION rather than by copy-paste, so it cannot drift from
--    whatever R's row actually says at apply time. The inheritance is a good
--    fit rather than a formality: R's boundaries field IS the headless release
--    scope Kevin granted on 07-26 (re-run gates from a clean origin/dev
--    worktree, no `railway`, no direct push to dev/main, no force-push, no
--    feature work, no improvised fixes, abort loudly on a red gate).
--
--    The ONE substitution: R's prompt_template opens "You are R·auto", and the
--    dispatcher composes its own "You are {letter}·auto" header from the row's
--    letter — so an inherited copy would hand R-A two contradicting identities
--    the moment prompt_template is consumed (Spec_Agents_Registry Phase 2; it
--    is display-only today). replace() still reads R's live row, so the
--    no-drift property holds; it substitutes the letter and nothing else.
INSERT INTO agents (letter, name, kind, department, status,
                    scope, boundaries, worktree, context_docs,
                    prompt_template, notes)
SELECT 'R-A', 'Release Executor', 'agent', r.department, 'headless',
       r.scope, r.boundaries, r.worktree, r.context_docs,
       replace(r.prompt_template, 'R·auto', 'R-A·auto'),
       'Headless half of the R lane (board #229, Kevin 07-30). Executes ONE '
       'release train against a brief: merge -> gate -> deploy-watch -> log. '
       'Runs gates twice and verifies before merging; ABORTS AND HANDS BACK '
       'rather than improvising; reports deploy state honestly and never stops '
       'at `pending`. Does NOT: decide what ships together or in what order, '
       'author or adapt a brief, reconcile a red gate or a conflict needing '
       'judgement, call an incident, or hold the cross-train picture (wave '
       'numbering, what supersedes what) - those stay with the R session, '
       'which is also the fallback when automation fails.'
FROM agents r
WHERE r.letter = 'R'
ON CONFLICT (letter) DO NOTHING;

-- 2. R becomes the live SESSION — undispatchable by construction, exactly like
--    `kevin` and (since #222) `M`. HOLD THIS UNTIL AFTER #229 step 3.
UPDATE agents
   SET status = 'live-session'
 WHERE letter = 'R'
   AND status <> 'live-session';
