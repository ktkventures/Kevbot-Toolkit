-- Seed the Dev Task Tracker with the current landscape (2026-06-19).
-- Run AFTER dev_tasks_table.sql. Safe to run once. Priority = phase.seq
-- (Phase 1 = MVP: reliable live trading + mass backtesting + something
-- profitable; Phase 2 = polish/scale). Lower seq = do sooner.

INSERT INTO dev_tasks
  (title, status, priority_phase, priority_seq, impacts_live, needs_live_validation, is_urgent, area, assignee, notes)
VALUES
  -- ── Phase 1 · MVP — most urgent at top ──────────────────────────────
  ('Monday: confirm Bug 5 live — coarse gates (313/312, 1d/4h) fire + no live↔backtest divergence',
     'Todo', 1, 1.1, true, true, true, 'engine', 'kevin',
     'Juneteenth blocked Fri validation. Rollback = flip RORT_TF_SCALED_WARMUP=0 on Worker. Commits 0e19838,3159d96. Watch railway logs --service Worker | grep [Bug5].'),
  ('Monday: confirm 1Min strategy warms + fires (no "Cannot resample to 1Min")',
     'Todo', 1, 1.2, true, true, false, 'engine', 'kevin',
     'Fix 8c2fdcb (<=60s native). Strat 194 TSLL/1Min was being skipped.'),
  ('310 RVOL live sub-bug — forming >60s bar gets ~2x volume → live RVOL gate never matches',
     'Backlog', 1, 1.3, true, false, false, 'engine', 'claude',
     'Suppress one volume source on the forming >60s bar. Verify Monday with live data.'),
  ('monitor_status stale / likely lingering engine instance (shows Apr-06/22-strats vs live 15:27/68)',
     'Backlog', 1, 1.4, true, false, false, 'infra', 'claude',
     'Status writer heartbeats with engine._start_time + len(strategies) but row is stale → suspect duplicate/zombie engine instance. May relate to WS connection trouble. Investigate Monday on a clean restart.'),
  ('First-UND / cold-seed is slow (~7min/strategy) — algo-lane cold-seed (#26)',
     'Backlog', 1, 1.5, false, false, false, 'backtest', 'claude',
     'Bottleneck is the algo lane, not backtest (Tier 2 dough does not touch it). Needs a product call: should a fresh strategy''s algo lane cold-seed historically or start empty + fill live?'),
  ('Stage-2: forward_test_start one source of truth (top-level col vs config dual-storage)',
     'Backlog', 1, 1.6, false, false, false, 'backend', 'claude', 'Bug 3. Landmine that confused the Fwd count.'),
  ('Stage-2: stamp cache-lane model (cache_None → cache_<algo_model>)',
     'Backlog', 1, 1.7, false, false, false, 'data', 'claude', 'Bug 4. None model id risks snapshot-mismatch re-catchup.'),
  ('Stage-2: distinguish backtest-seed vs live equity on the Forward line (viz)',
     'Backlog', 1, 1.8, false, false, false, 'frontend', 'claude', 'Bug 2. cache lane starts as a backtest seed-copy.'),
  ('rsi_zones / rsi_zones_3 template-not-found warnings (an older strategy refs a removed pack)',
     'Backlog', 1, 1.9, false, false, false, 'data', 'claude', 'Benign (skips group) but floods logs. New strats 312-331 are all swing_123, no RSI.'),

  -- ── Phase 2 · polish / scale ────────────────────────────────────────
  ('Tier 2 into the UND / cold-seed path (fast first-update via dough re-bake)',
     'Backlog', 2, 2.1, false, false, false, 'backtest', 'claude', 'Depends on the 1.5 algo-lane product call. Tier 2 already live for mode=all recomputes.'),
  ('Divergence draggers: B4 1Min early-fire (#5, 265/269), 292 SR-channels, 275 dead-live, 271 5m under-pass',
     'Backlog', 2, 2.2, true, false, false, 'engine', 'claude', ''),
  ('#16 — guard full-UAD so a transient fetch failure cannot silently zero a lane',
     'Backlog', 2, 2.3, false, false, false, 'backtest', 'claude', ''),
  ('#30 — pin the Fidelity Gate fixture window (deterministic golden)',
     'Backlog', 2, 2.4, false, false, false, 'infra', 'claude', ''),
  ('Trade-engine throughput (process_bar = 76% of a recompute) — the real recompute lever',
     'Backlog', 2, 2.5, false, false, false, 'engine', 'claude', 'Risky/big; separate decision.'),
  ('gap-healer holiday calendar (expects bars on market holidays e.g. Juneteenth)',
     'Backlog', 2, 2.6, false, false, false, 'infra', 'claude', 'Cosmetic — correctly stays absent, but logs noise.'),
  ('Docs cleanup phase 2 — move active docs into _active/, archive done specs',
     'Backlog', 2, 2.7, false, false, false, 'docs', 'claude', ''),

  -- ── Recently shipped (reference; hide-Done to clear) ────────────────
  ('Tier 2 dough cache — validated byte-identical vs full recompute, ~60x, live (RORT_USE_DOUGH_CACHE=1)',
     'Done', 1, 1.05, false, false, false, 'backtest', 'claude', 'Stable key + visible-trim fixes; dry-run sid 318 byte-identical.'),
  ('Tier 1 — MB save auto-populates backtest lane (recompute when trades-less)',
     'Done', 1, 1.04, false, false, false, 'backend', 'claude', ''),
  ('Bug 1 + equity OOS alignment — Fwd count = backtest-lane @ in_sample_end',
     'Done', 1, 1.03, false, false, false, 'frontend', 'claude', ''),
  ('#21 prep scoping + #29 chart speed (byte-identical, OOM relief)',
     'Done', 1, 1.02, false, false, false, 'backtest', 'claude', ''),
  ('Dev Task Tracker (this board)',
     'Done', 1, 1.01, false, false, false, 'frontend', 'claude', 'phase.seq priority, impacts_live/needs_validation badges, comments.');
