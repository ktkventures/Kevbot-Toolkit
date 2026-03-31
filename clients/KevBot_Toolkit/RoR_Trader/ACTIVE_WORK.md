# RoR Trader — Active Work Tracker

**Last Updated:** 2026-03-31
**Current Phase:** Phase 1 (Engine LC/CC + Reference Packs) — COMPLETE
**Current Focus:** Strategy Builder polish before moving to Phase 2

---

## Phase 1: Engine LC/CC Support + Reference Packs — COMPLETE

All subtasks (1a-1i) completed in earlier sessions:
- Engine LC/CC execution types fully wired + tested
- EMA Price Position V2 + Swing 1-2-3 reference packs
- Migration guard + regression tests

---

## Strategy Builder — Completed This Session

### Immediate Next Items (steps 1-6) — ALL DONE
- [x] 1. Exit cross markers (+) at exit price levels
- [x] 2. C-type timestamp shift on chart markers (plot entries/exits on next bar open)
- [x] 3. TF Conditions showing all timeframes (secondary_tfs in backtest request)
- [x] 4. Confluence Depth auto-search (find_best_combinations wired to depth selector 1/2/3/4)
- [x] 5. Progress bar (indeterminate — actual scenario counts deferred, needs SSE)
- [x] 6. Filter/sort applied to ALL tabs (was only Entry + TF Conditions)

### Also Completed This Session
- [x] All 6 analysis tabs wired (Entry, Exit, TF Conditions, General, Stop Loss, Take Profit)
- [x] Price chart upgrade (SyncedChartPane with overlays, oscillators, heatmap, trade markers)
- [x] Equity curve per-trade + per-day modes (HWM merge fix)
- [x] General tab confluence depth 2+ (include_prefix passed to find_best_combinations)
- [x] Stale results bug (sibling key clearing on depth switch)
- [x] inf/nan sanitization in all KPI serialization
- [x] SyntaxError crash fix in _analyze_combinations_impl

---

## Strategy Builder — Noted for Later (Not Yet Started)

These were identified during testing but deferred:

- [ ] **Stop/Take Profit pack parameter separation** — UI shows mixed stop+target params, needs frontend filtering
- [ ] **Strategy Builder defaults settings page** — user-configurable default entry, exit, conditions, stop, target
- [ ] **Legacy pack cleanup** — remove LC/CC exec variants from legacy templates, tag properly
- [ ] **Swing 1-2-3 not returning backtest data** — needs investigation
- [ ] **Trade qualification feature in filter/sort modal** — design exists, backend not built
- [ ] **TF Conditions fidelity badge** — should show PB/CB instead of C/L exec type
- [ ] **Price chart legend** — add show/hide toggles for conditions and triggers (like Strategy Detail)
- [ ] **Progress tracking with actual scenario counts** — needs SSE for real progress streaming

---

## 6-Phase Roadmap

### Phase 1: Engine LC/CC Support + Reference Packs — COMPLETE
All subtasks 1a-1i done. Engine handles C, L, LC, CC execution types.

### Phase 2: Strategy Detail Page Refinements — NOT STARTED
- [ ] 2a. Full/short trigger names with pack origin (e.g., "UT Bot v2 [Aggressive]: Buy")
- [ ] 2b. PB/CB per-confluence breakdown using confluence_records field
- [ ] 2c. Date range selector (state exists, needs wiring)
- [ ] 2d. Exec type color coding for [L], [LC], [CC]

### Phase 3: Hi-Fi Execution & Hold Times — NOT STARTED
- [ ] 3a. Ambiguous bar detection (stop + target both in bar's range)
- [ ] 3b. 1-second bar loading from Polygon for ambiguous resolution
- [ ] 3c. Hold time computation (hold_time_seconds, hold_bars in trade records)
- [ ] 3d. Surface hold times in trades table + KPIs

### Phase 4: Chart Test Page + Replay — NOT STARTED
- [ ] 4a. Page scaffold with chart + config panel
- [ ] 4b. Configuration: symbol, timeframe, date range, trigger selector (C/L/LC/CC), stop/target
- [ ] 4c. Instant backtest on config change
- [ ] 4d. Replay feature (TradingView-style bar-by-bar playback)
- [ ] 4e. Trade markers with exec type colors, stop/target price lines
- [ ] 4f. Lock chart range toggle, copy-to-strategy-builder

### Phase 5: Pack Builder Update + Swing 123 / Golden Candle — NOT STARTED
- [ ] 5a. Extend pack_spec.py validation for exec_variants
- [ ] 5b. Update pack_builder_context.md with exec_variants docs + examples
- [ ] 5c. Update pack_builder.py LLM prompt
- [ ] 5d. Update pack_registry.py to register LC/CC triggers
- [ ] 5e. First Pack Builder test: generate a pack with LC/CC
- [ ] 5f. Backward compat: exec_variants optional, existing packs default C-only

### Phase 6: Alert Monitor & Webhook Updates — NOT STARTED
- [ ] 6a. Add exec_type to alert records + new placeholder tokens
- [ ] 6b. L-type exit timing awareness
- [ ] 6c. Account-level webhook templates
- [ ] 6d. Exec-type-aware event filtering

---

## Key Decisions (Reference)

- LC is a superset of HM — configurable confirm_bar_offset + bail_action vs HM's hardcoded same-bar + market bail
- CC is genuinely new — close-to-close with next-bar confirmation
- Suffix-based dispatch — _lc/_cc suffixes, no database migration
- Two reference packs — EMA PP V2 (known baseline) + Swing 123 (natural CC fit)
- Replay in Phase 4 — saves days of live-data waiting for pack validation
- HM/HL backward compat — continue working, eventually LC with offset=0 + bail=market is equivalent to HM
