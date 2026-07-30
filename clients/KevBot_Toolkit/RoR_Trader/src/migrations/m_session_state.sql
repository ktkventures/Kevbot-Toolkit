-- M-Session Dashboard — the ONE genuinely new store (2026-07-30, board #220
-- step 4).  ⛔ AUTHORED, NOT APPLIED. Kevin authorizes by name before this runs.
--
-- Design principle B from the task, and the reason this file is three narrow
-- tables rather than one generic key/value blob:
--
--   "IT IS A VIEW, NOT A SECOND BOARD. Sections 1 and 2 read dev_tasks /
--    dev_task_comments / run_history. The ONLY new stored state is M-specific
--    and cannot be derived: M's intended ordering, M's seen/actioned marks, and
--    M's canvas text. If it starts storing task state, it will drift from the
--    board and Kevin will have two answers to the same question."
--
-- A generic `m_session_state(key, value JSONB)` would have satisfied every
-- requirement here and quietly become the second board — nothing would stop a
-- future writer parking a status or an assignee in it. These tables are shaped
-- so that storing a board-owned field would require ANOTHER migration, i.e. it
-- would be visible. That is the whole point: every column below is something
-- the board cannot answer.
--
-- Additive: three NEW tables, nothing existing changes. Same posture as
-- dev_tasks / run_history / task_mentions — admin-only via the service-role
-- client, RLS on with no public policy.


-- ── 1 · M's canvas ────────────────────────────────────────────────────────
-- Kevin, 07-30: "a good canvas for you to publish the details and I'll take a
-- look at it" — the fix for the read-it-twice problem. TASK-SPECIFIC detail
-- stays on the task thread (its audit trail); the canvas carries the
-- CROSS-TASK picture: what M is doing, worried about, waiting on.
--
-- Singleton by construction (CHECK id = 1): there is exactly one M session and
-- therefore exactly one canvas. A table that could hold two canvases would
-- immediately raise "which one is current?", which is the question this store
-- exists to remove.
CREATE TABLE IF NOT EXISTS m_session_canvas (
    id          INT         PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    body        TEXT        NOT NULL DEFAULT '',
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by  TEXT                                  -- 'M' | 'M·auto' | 'kevin'
);

-- Seed the singleton so a read never has to distinguish "empty canvas" from
-- "no row yet" — two states that render identically and debug differently.
INSERT INTO m_session_canvas (id, body, updated_by)
VALUES (1, '', 'migration')
ON CONFLICT (id) DO NOTHING;


-- ── 2 · M's intended ordering ─────────────────────────────────────────────
-- Kevin: "the order M INTENDS, which is a plan, not board metadata. Today that
-- plan lives only in M's head and a session-local todo list Kevin cannot see."
-- That is precisely why it needs a row: it is NOT derivable from priority_phase
-- / priority_seq, which say what the BOARD thinks the priority is.
--
-- Deliberately NOT stored here: status, assignee, title, priority_phase,
-- priority_seq, or anything else `dev_tasks` owns. The row is (task_id, rank,
-- why) and nothing more. A task not in this table simply falls back to board
-- order — absence is a valid, meaningful state, not a gap to be backfilled.
CREATE TABLE IF NOT EXISTS m_session_queue_order (
    task_id     BIGINT      PRIMARY KEY REFERENCES dev_tasks(id) ON DELETE CASCADE,
    -- Position in M's plan, ascending. Sparse and non-contiguous on purpose:
    -- re-ranking one item must not rewrite the whole table.
    rank        INT         NOT NULL,
    -- Optional one-liner: WHY it sits there. This is the field that answers
    -- Kevin's "why is that third?" without him having to ask.
    why         TEXT,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by  TEXT
);

-- The only read is "the whole plan, in order".
CREATE INDEX IF NOT EXISTS idx_m_session_queue_order_rank
    ON m_session_queue_order (rank);


-- ── 3 · M's seen / actioned marks ─────────────────────────────────────────
-- The accountability artifact: "event -> M saw it -> M did X, or judged no
-- action needed". A coverage record, not a log.
--
-- Keyed by the FEED EVENT KEY (`comment:41`, `run:7`, and from Phase 2 `env:*`)
-- and NOT by a comment id, because roughly half of what M actually watches is
-- not in the database at all — stranded branches, loop-version drift, preflight
-- reds. A FK to dev_task_comments here would weld the coverage record to
-- `dev_task_comments` and make Phase 2 a migration instead of a writer. TEXT
-- key, no FK, on purpose.
--
-- `task_id` is a nullable convenience for "show me coverage on this task" — an
-- environment event has no task, which is the case a NOT NULL would have
-- forbidden.
CREATE TABLE IF NOT EXISTS m_session_event_marks (
    event_key   TEXT        PRIMARY KEY,
    -- 'seen'     = M looked and the rule's verdict stands
    -- 'actioned' = M did something; `note` says what
    -- 'no-action'= M looked and judged nothing was needed; `note` says why
    state       TEXT        NOT NULL CHECK (state IN ('seen', 'actioned', 'no-action')),
    -- Exceptions get a real note; routine confirmations do not need one. The
    -- column is nullable because forcing prose on every row is the theatre the
    -- task explicitly rejected.
    note        TEXT,
    task_id     BIGINT      REFERENCES dev_tasks(id) ON DELETE CASCADE,
    marked_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    marked_by   TEXT
);

CREATE INDEX IF NOT EXISTS idx_m_session_event_marks_task
    ON m_session_event_marks (task_id, marked_at DESC);


ALTER TABLE m_session_canvas       ENABLE ROW LEVEL SECURITY;
ALTER TABLE m_session_queue_order  ENABLE ROW LEVEL SECURITY;
ALTER TABLE m_session_event_marks  ENABLE ROW LEVEL SECURITY;
-- No public policies: only the service-role (admin) client reaches these
-- tables, via the admin-gated /api/m-session endpoints. Service role bypasses
-- RLS. Same posture as dev_tasks.
