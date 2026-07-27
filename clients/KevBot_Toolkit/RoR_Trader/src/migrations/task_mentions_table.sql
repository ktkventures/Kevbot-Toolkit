-- Mentions + unread Messages (2026-07-27, board #143) — close the last gap
-- where a comment one teammate leaves the other never circles back to.
-- ADDITIVE: one NEW table only; nothing existing changes. Written at
-- comment-POST time by the dev_tasks API (parse @tokens against the agents
-- registry letters + kevin). Same posture as dev_tasks / agents: admin-only
-- via the service-role client, RLS on with no public policy.

CREATE TABLE IF NOT EXISTS task_mentions (
    id          BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    -- the comment that carried the @token; drops with its comment/task
    comment_id  BIGINT      NOT NULL REFERENCES dev_task_comments(id) ON DELETE CASCADE,
    task_id     BIGINT      NOT NULL REFERENCES dev_tasks(id) ON DELETE CASCADE,
    -- canonical registry token being mentioned: 'M' | 'E' | 'E2' | 'F' | 'P'
    -- | 'R' | 'kevin' (case-normalized to the agents.letter it matched)
    mentioned   TEXT        NOT NULL,
    -- who wrote the comment (the mention's sender)
    author      TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- NULL = unseen (the inbox); stamped when the mention is delivered/opened
    seen_at     TIMESTAMPTZ,
    -- One row per (comment, mentioned) — a comment tags each role at most once.
    -- Makes the parser and the optional backfill idempotent (re-run safe).
    UNIQUE (comment_id, mentioned)
);

-- The hot query is the inbox: "unseen mentions for role R" and "recent seen
-- for role R" — both lead with mentioned, then seen_at.
CREATE INDEX IF NOT EXISTS idx_task_mentions_inbox
    ON task_mentions (mentioned, seen_at);

ALTER TABLE task_mentions ENABLE ROW LEVEL SECURITY;
-- No public policies: only the service-role (admin) client reaches this table,
-- via admin-gated API endpoints. Service role bypasses RLS.

-- ── Optional one-time backfill (spec: "nice-to-have if cheap") ────────────
-- Seed mentions from the last 7 days of comments so the feature lands with
-- history instead of an empty inbox. Idempotent (ON CONFLICT DO NOTHING) and
-- safe to omit. Postgres regex has no lookbehind, so we consume a leading
-- boundary char — (^|non-alnum-non-@) — to avoid matching email local parts
-- like foo@bar; the JOIN to agents drops any @token that isn't a real role.
INSERT INTO task_mentions (comment_id, task_id, mentioned, author, created_at)
SELECT c.id, c.task_id, a.letter, c.author, c.created_at
FROM dev_task_comments c
CROSS JOIN LATERAL regexp_matches(
    c.body, '(^|[^[:alnum:]@])@([A-Za-z][A-Za-z0-9]*)', 'g') AS m
JOIN agents a ON lower(a.letter) = lower(m[2])
WHERE c.author <> 'system'
  AND c.created_at > now() - interval '7 days'
ON CONFLICT (comment_id, mentioned) DO NOTHING;
