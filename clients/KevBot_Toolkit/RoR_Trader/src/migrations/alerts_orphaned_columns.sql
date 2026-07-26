-- ============================================================================
-- Alerts orphaned-annotation columns — schema migration
-- 2026-07-26 — Board #117 (V1.14) dangling alert-history "Open" rows
--
-- Adds two additive, nullable, top-level columns to the alerts table:
--
--   orphaned_at      TIMESTAMPTZ  -- when the annotation sweeper marked this
--                                    unpaired entry_signal as orphaned. NULL =
--                                    not annotated (the normal case).
--   orphaned_reason  TEXT         -- why it is orphaned. One of:
--                                    'redundant_entry'      (a later entry_signal
--                                       superseded this one while already in
--                                       position; LIFO orphaned the earlier row)
--                                    'redundant_entry_slow' (same, >1h gap)
--                                    'eod_churn'            (fired in the 19:54-
--                                       20:06Z end-of-day churn window)
--                                    'lost_exit'            (no exit alert ever
--                                       fired; reserved — the sweeper does NOT
--                                       auto-set this class, see below)
--
-- WHY THIS EXISTS
--   Alert-lane "Open" is NOT a stored DB state. The alerts table is one row per
--   EVENT (type='entry_signal' / 'exit_signal'); there is no exit_time / status
--   / is_open column. "Open" is DERIVED client-side by LIFO-pairing entries to
--   exits (frontend/src/views/StrategyDetailPage.tsx). A leftover unpaired
--   entry_signal renders result:'Open' forever = a "dangling open" position that
--   is not actually held.
--
--   Kevin's hard rule (board #117): alert history must stay TRUE to when alerts
--   actually fired. We NEVER synthesize a retroactive exit_signal (firing a sell
--   days later would push a live order for a position no longer held). So the
--   only correct closure is an ANNOTATION on the leftover entry row itself — the
--   row's type / timestamp / price / direction are left untouched, and no
--   exit_signal is ever inserted.
--
--   The UI reads orphaned_at: a non-null value renders the leftover as a muted
--   "⚠ Orphaned — <reason>" instead of a live "Open" position (frontend lane, F).
--
-- WHO WRITES THESE
--   src/annotate_orphaned_alerts.py — an offline, idempotent, dry-run-first
--   sweeper. It sets orphaned_at = now() + orphaned_reason on STALE unpaired
--   entry_signal rows of the provable display-artifact classes only
--   (redundant_entry / redundant_entry_slow / eod_churn). It intentionally does
--   NOT annotate 'lost_exit' or anything recent enough to be a genuinely-open
--   live position — those route to position-state reconciliation (board #114).
--
-- No backfill here — the sweeper does the (reviewable, reversible) annotation.
-- Legacy alerts stay NULL until the sweeper runs.
--
-- Run in Supabase SQL Editor. Idempotent.
-- ============================================================================

ALTER TABLE alerts
  ADD COLUMN IF NOT EXISTS orphaned_at TIMESTAMPTZ;

ALTER TABLE alerts
  ADD COLUMN IF NOT EXISTS orphaned_reason TEXT;

-- Small partial index for the "count / list annotated" lookups (UI badge,
-- sweeper idempotency verify). Only indexes the annotated minority.
CREATE INDEX IF NOT EXISTS alerts_orphaned_at_idx
  ON alerts (orphaned_at)
  WHERE orphaned_at IS NOT NULL;

-- Verify after running:
-- SELECT
--   orphaned_reason,
--   count(*) AS n,
--   min(timestamp) AS earliest,
--   max(timestamp) AS latest
-- FROM alerts
-- WHERE orphaned_at IS NOT NULL
-- GROUP BY orphaned_reason
-- ORDER BY n DESC;

-- Rollback (fully reversible — additive nullable columns, sweeper writes only):
-- To un-annotate everything the sweeper stamped:
--   UPDATE alerts SET orphaned_at = NULL, orphaned_reason = NULL
--   WHERE orphaned_at IS NOT NULL;
-- To drop the columns entirely:
--   DROP INDEX IF EXISTS alerts_orphaned_at_idx;
--   ALTER TABLE alerts DROP COLUMN IF EXISTS orphaned_reason;
--   ALTER TABLE alerts DROP COLUMN IF EXISTS orphaned_at;
