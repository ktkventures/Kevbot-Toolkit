-- Board #261 — explicit task types on dev_tasks.
--
-- WHY: the board already behaves as though tasks have types — `isVisionTask()`
-- derives "vision" from `tags @> {vision}` OR "has children", and VISION_STATUSES
-- already gives visions their own status set — but the type itself is implicit and
-- the status split is frontend-only. This makes the type a first-class field so it
-- can govern statuses, icons and rules (Kevin's ruling, 2026-07-31: an explicit
-- column, not a tag, because behaviour keyed off a free-text tag lies the first
-- time someone writes 'Vision' or 'goals').
--
-- `goal_params` ships in THIS file on purpose. It is consumed by board #262 (the
-- goal task type), not by #261. Including it here means Kevin authorizes one
-- migration instead of two for the same arc; it is additive, nullable, and nothing
-- reads it until #262 lands.
--
-- SAFETY:
--   * Additive only. No column is altered or dropped; no row loses data.
--   * Idempotent (IF NOT EXISTS + a guarded UPDATE) — safe to re-run.
--   * The backfill reproduces TODAY'S derivation EXACTLY, so no task changes
--     behaviour on the day this lands. Measured against prod 2026-07-31:
--         236 tasks total
--           6 tagged 'vision'
--           6 have children
--           7 in the UNION  -> task_type = 'vision'
--         229 remaining     -> task_type = 'action'
--     (5 rows are both tagged and have children; 1 is tagged only; 1 has children
--      only — which is precisely why the rule is a UNION and not either half.)
--   * No existing container is itself a child, so this backfill cannot nest
--     containers.
--
-- NOTE ON `tags`: it is `text[]`, NOT jsonb. The containment operator below is the
-- array form. A first draft of this file used jsonb containment and would have
-- failed on apply — checked against information_schema rather than assumed.
--
-- AUTHORIZATION: authored by M, 2026-07-31. NOT to be applied by any agent.
-- Kevin authorizes by name (board #261 step 2), then M or Kevin applies, and only
-- then may code reading `task_type` merge.

BEGIN;

ALTER TABLE dev_tasks
  ADD COLUMN IF NOT EXISTS task_type text NOT NULL DEFAULT 'action';

ALTER TABLE dev_tasks
  ADD COLUMN IF NOT EXISTS goal_params jsonb NULL;

-- Keep the vocabulary honest at the database boundary. Three types today; adding a
-- fourth (e.g. 'loop', which Kevin raised and deferred) means altering this
-- constraint deliberately rather than discovering a typo in production.
ALTER TABLE dev_tasks
  DROP CONSTRAINT IF EXISTS dev_tasks_task_type_check;

ALTER TABLE dev_tasks
  ADD CONSTRAINT dev_tasks_task_type_check
  CHECK (task_type IN ('action', 'vision', 'goal'));

-- Backfill: EXACTLY today's `isVisionTask()` — tagged 'vision' OR already a parent.
-- Guarded on the default so a re-run cannot demote a type set deliberately later.
UPDATE dev_tasks
   SET task_type = 'vision'
 WHERE task_type = 'action'
   AND (
         tags @> ARRAY['vision']::text[]
         OR id IN (SELECT DISTINCT parent_id FROM dev_tasks WHERE parent_id IS NOT NULL)
       );

COMMIT;

-- POST-APPLY VERIFICATION (board #261 step 3 — the set-equality check, not a count):
--
--   SELECT task_type, count(*) FROM dev_tasks GROUP BY task_type ORDER BY task_type;
--     -- expect: action 229, vision 7   (as measured 2026-07-31)
--
--   -- Every row marked 'vision' must be one today's rule would return, and vice
--   -- versa. This must return ZERO rows. A count that matches while the SETS
--   -- differ would mean the backfill moved the wrong tasks.
--   SELECT id, task_type, tags, (id IN (SELECT parent_id FROM dev_tasks
--                                        WHERE parent_id IS NOT NULL)) AS has_kids
--     FROM dev_tasks
--    WHERE (task_type = 'vision')
--       <> (tags @> ARRAY['vision']::text[]
--           OR id IN (SELECT parent_id FROM dev_tasks WHERE parent_id IS NOT NULL));
--
--   SELECT count(*) FROM dev_tasks WHERE goal_params IS NOT NULL;  -- expect 0
