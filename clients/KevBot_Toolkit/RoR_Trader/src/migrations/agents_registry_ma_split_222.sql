-- Agents Registry — split `M` into M (session) + M-A (manager assistant).
-- Board #222, step 2 (F) · 2026-07-30 · design: Kevin, 07-30.
--
-- DATA, NOT SCHEMA. No DDL: both statements are ordinary INSERT/UPDATE against
-- columns `agents` has had since agents_registry.sql (2026-07-23). The `status`
-- vocabulary already contains 'live-session' — it is what `kevin` and `E2` sit
-- on today, and the whole point of this change is that the mechanism is already
-- built and proven rather than newly invented.
--
-- WHY
-- `M` names two different actors: the long-running manager SESSION (holds the
-- day's context, talks to Kevin, reviews other agents' work, restarts the loop)
-- and the headless infrastructure engineer that builds the team's own tooling.
-- The board could not tell them apart, so twice on 07-30 (#195, #219) a task
-- assigned `M` and left ARMED dispatched `M·auto` into work reserved for the
-- session — one of those steps literally read "M restarts the loop".
--
-- HOW IT FIXES IT — no new mechanism
-- `dispatcher.headless_agents()` selects `status=eq.headless`. Flipping `M` to
-- 'live-session' drops it out of that set, and `triage_todo()` already has the
-- correct behaviour for an assignee that is not a headless agent: skipped with
-- `is_stuck=False` — "waiting on M (not a headless agent)", quiet, no staleness
-- tripwire. That is the designed state, and `kevin` has proven it since 07-23.
--
-- ORDERING (deliberate — read before applying)
--   Statement 1 (INSERT M-A) is ADDITIVE and safe to apply at any time: it adds
--   a lane nothing is assigned to yet.
--   Statement 2 (UPDATE M) is the BEHAVIOUR CHANGE. The moment it lands, every
--   open task assigned `M` stops dispatching. Board #222 step 3 (M classifies
--   all 16 open tasks and 28 incomplete steps as M or M-A) exists precisely so
--   that flip does not strand work that should have been M-A's. Apply
--   statement 2 with the release train (step 4), AFTER the classification.
--
-- Idempotent: re-running is a no-op.

-- 1. M-A inherits M's scope, boundaries, worktree, context docs and prompt
--    template BY SELECTION rather than by copy-paste, so it cannot drift from
--    whatever M's row actually says at apply time.
INSERT INTO agents (letter, name, kind, department, status,
                    scope, boundaries, worktree, context_docs,
                    prompt_template, notes)
SELECT 'M-A', 'Manager Assistant', 'agent', m.department, 'headless',
       m.scope, m.boundaries, m.worktree, m.context_docs,
       m.prompt_template,
       'Headless half of the M lane (board #222, Kevin 07-30). Builds M-lane '
       'infrastructure, writes specs/design docs, runs sweeps and audits, '
       'diagnoses bounded defects, verifies a release train''s claims. Runs in '
       'PARALLEL on fresh bounded context and leaves a run_history row. Does '
       'NOT: review another agent''s work, reconcile a merge needing judgement, '
       'restart the loop, apply a migration, touch prod, set priority, talk to '
       'Kevin, or author a chain — those stay with the M session.'
FROM agents m
WHERE m.letter = 'M'
ON CONFLICT (letter) DO NOTHING;

-- 2. M becomes the live SESSION — undispatchable by construction, exactly like
--    `kevin`. HOLD THIS UNTIL AFTER #222 step 3 (see ORDERING above).
UPDATE agents
   SET status = 'live-session'
 WHERE letter = 'M'
   AND status <> 'live-session';
