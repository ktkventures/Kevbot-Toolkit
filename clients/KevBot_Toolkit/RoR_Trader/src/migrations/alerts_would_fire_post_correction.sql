-- Migration: alerts.would_fire_post_correction — separates structural-WS
-- phantoms from real-bug phantoms.
-- Apply via Supabase SQL editor.
--
-- Why: ws_rest_spliced corrects bar OHLCV after a splice fires, but the
-- alert that triggered on the WS-derived close already went out. The
-- distinction between "real divergence" and "WS-vs-REST point-in-time
-- disagreement" was previously invisible — both showed up as phantoms.
-- After this migration:
--   * NULL                — no splice fired on this alert's bar (no info)
--   * TRUE  ('would_fire')— REST-corrected indicator state STILL fires the
--                           same trigger → alert is real, phantom-status
--                           is a different cause (probably a backtest gap)
--   * FALSE ('would_not_fire')— REST-corrected state would NOT have fired
--                           the trigger → phantom is structural WS noise,
--                           reclassify away from "needs_investigation"
--
-- strategy_health classifier uses this column to surface a new
-- `phantom_pre_correction` bucket (alongside cross_exec_type_mismatch,
-- phase2_signal_exit, etc.) so divergence reporting is honest.

ALTER TABLE alerts
    ADD COLUMN IF NOT EXISTS would_fire_post_correction BOOLEAN;

CREATE INDEX IF NOT EXISTS alerts_would_fire_post_correction_idx
    ON alerts (would_fire_post_correction)
    WHERE would_fire_post_correction IS NOT NULL;

COMMENT ON COLUMN alerts.would_fire_post_correction IS
    'Set by ralph_engine.apply_rest_correction after a REST splice fires '
    'on this alert''s bar. TRUE = trigger still fires given REST-aligned '
    'state (alert is real). FALSE = trigger would not fire = '
    'phantom_pre_correction (structural WS noise). NULL = no splice '
    'happened on this bar, classification unchanged.';
