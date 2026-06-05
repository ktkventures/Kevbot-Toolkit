-- Migration: bar_diagnostics — add 'live_corrected' source value
-- Apply via Supabase SQL editor.
--
-- Purpose: separate the pre-correction (instant) and post-correction
-- (settled) snapshots of live indicator state. Pre = what the engine
-- emitted at bar close (source='live'); post = what the engine
-- converged to after rest_verifier splice (source='live_corrected').
--
-- Why both: the 'live' snapshot is the real-time truth (what the engine
-- actually used to fire signals at that moment); the 'live_corrected'
-- snapshot is the REST-aligned reality used for backtest comparison.
-- Pair-rate analysis uses 'live_corrected' vs 'backtest' for the
-- apples-to-apples comparison. Divergence analysis uses 'live' to see
-- the worst-case state the engine actually saw before correction.
--
-- The Postgres CHECK constraint can't be modified in place — drop and
-- recreate with the expanded set.

ALTER TABLE bar_diagnostics
    DROP CONSTRAINT bar_diagnostics_source_check;

ALTER TABLE bar_diagnostics
    ADD CONSTRAINT bar_diagnostics_source_check
    CHECK (source IN ('live', 'live_corrected', 'backtest', 'algo'));
