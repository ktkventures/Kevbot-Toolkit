# RoR Trader — Active Work Tracker

**Last Updated:** 2026-03-31
**Current Phase:** Phase 2 (Strategy Detail Page Refinements) — IN PROGRESS
**Current Focus:** Phase 2 implementation complete, pending QA verification

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

## Strategy Builder — Noted for Later (Triaged → Assigned to Phases)

Items identified during testing, now assigned to roadmap phases:

- **Stop/Take Profit pack restructuring + parameter separation** → Phase 2 (item 2a-new)
- **Strategy Builder defaults (entry, exit, stop)** → Phase 2 (item 2b-new)
- **Legacy pack cleanup (mark as legacy, keep functional)** → Phase 5 (item 5a-new)
- **Swing 1-2-3 — rebuild via Pack Builder as first test case** → Phase 5 (item 5e)
- **Trade qualification filter in Strategy Builder** → Phase 5 (item 5g-new, needs backend first)
- **TF Conditions fidelity badge (PB/CB)** → Phase 3 (item 3e-new, depends on Hi-Fi)
- ~~**Price chart legend show/hide toggles**~~ → CANCELED (current behavior matches Strategy Detail, preferred as-is)
- **Progress tracking with SSE scenario counts** → Phase 4 (item 4g-new, SSE infra benefits Chart Test too)

---

## 6-Phase Roadmap

### Phase 1: Engine LC/CC Support + Reference Packs — COMPLETE
All subtasks 1a-1i done. Engine handles C, L, LC, CC execution types.

### Phase 2: Strategy Detail Page Refinements — IN PROGRESS

**Priority items (from Strategy Builder triage):**
- [x] 2a-new. **Stop/Take Profit pack restructuring** — Added stop_params/target_params to all 7 RM TEMPLATES. API now returns template_name, stop_summary, target_summary. Frontend uses dedicated useStopLossPacks()/useTakeProfitPacks() hooks with separated parameter display. Stop packs show only stop params, target packs show only target params.
- [x] 2b-new. **Strategy Builder defaults (entry, exit, stop)** — Added useStrategyBuilderDefaults Zustand store with localStorage persistence. Strategy Builder auto-populates from saved defaults on mount. "Save as Default" / "Clear Defaults" buttons in config section.

**Existing items:**
- [x] 2c. Full/short trigger names with pack origin — Changed trigger name format from "Group: Trigger" to "Group > Trigger". Strategy Builder now parses pack name from trigger name for grouped display.
- [x] 2d. PB/CB per-confluence breakdown — Backend already had enrich_confluence_with_fidelity() returning confluence_enriched. Wired to frontend — [PB]/[CB] badges now data-driven per condition instead of hardcoded. All current conditions show [PB]; cross-TF conditions will show [CB] after Hi-Fi (Phase 3).
- [x] 2e. Date range selector — Added date_range query param to GET /api/strategies/{id}. Backend filters stored_trades by date before enrichment. Frontend passes dateRange state to useStrategy hook, refetches on change. Supports: Last 7/30/90 Days, Backtest Only, Forward Only.
- [x] 2f. Exec type color coding — Fixed ExecBadge on both Strategy Detail and Strategy Builder to use distinct colors: [C] blue, [L] green, [LC] purple, [CC] orange. Was dead code before (execTypeColors defined but never used).

### Phase 3: Hi-Fi Execution & Hold Times — NOT STARTED
- [ ] 3a. Ambiguous bar detection (stop + target both in bar's range)
- [ ] 3b. 1-second bar loading from Polygon for ambiguous resolution
- [ ] 3c. Hold time computation (hold_time_seconds, hold_bars in trade records)
- [ ] 3d. Surface hold times in trades table + KPIs
- [ ] 3e-new. **TF Conditions fidelity badge (PB/CB)** — Update TF condition displays to show [PB] (previous bar) or [CB] (current bar) fidelity type instead of C/L exec type. CB uses Hi-Fi 1-second drilling to determine cross-TF condition state at entry time. Depends on 3a-3b being complete.

### Phase 4: Chart Test Page + Replay — NOT STARTED
- [ ] 4a. Page scaffold with chart + config panel
- [ ] 4b. Configuration: symbol, timeframe, date range, trigger selector (C/L/LC/CC), stop/target
- [ ] 4c. Instant backtest on config change
- [ ] 4d. Replay feature (TradingView-style bar-by-bar playback)
- [ ] 4e. Trade markers with exec type colors, stop/target price lines
- [ ] 4f. Lock chart range toggle, copy-to-strategy-builder
- [ ] 4g-new. **Progress tracking with SSE** — Server-Sent Events for real scenario counts during Strategy Builder analysis + Chart Test backtests. Replace indeterminate progress bar with "Scenario 47 of 312" style updates. SSE infrastructure benefits both pages.

### Phase 5: Pack Builder Update + Swing 123 / Golden Candle — NOT STARTED

**Priority items (from Strategy Builder triage):**
- [ ] 5a-new. **Legacy pack cleanup** — Mark all current TF Confluence packs (except EMA Price Position V2) as "Legacy (Default)" in their display names. Keep them fully functional — legacy packs use the engine's generic LC/CC wrapping which works but isn't purpose-built like V2. Legacy label stays until packs are rebuilt through Pack Builder or manually updated to V2's explicit trigger variant structure. Do NOT delete or disable legacy packs.

**Pack Builder pipeline (get EMA PP V2 perfect → inform Pack Builder → scale):**
- [ ] 5b. Extend pack_spec.py validation for exec_variants
- [ ] 5c. Update pack_builder_context.md with exec_variants docs + examples (modeled on EMA PP V2 as the reference)
- [ ] 5d. Update pack_builder.py LLM prompt (include flicker logic, painting logic, execution fidelity checks)
- [ ] 5e. Update pack_registry.py to register LC/CC triggers
- [ ] 5f. **Swing 1-2-3 as first Pack Builder test case** — Don't fix the current broken swing_123 manually. Instead, use it as the first real test of the Pack Builder pipeline. Validates that the system can scalably produce packs with proper CC execution, correct trigger detection, and accurate backtests.
- [ ] 5g. Backward compat: exec_variants optional, existing packs default C-only
- [ ] 5h-new. **Trade qualification filter in Strategy Builder** — Add TQ filter dropdown to the filter/sort modal. Requires backend TQ rule application logic to be built first. Design exists in Frontend Migration Plan (Portfolio Requirements V5 spec).

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
- **Golden Child approach** — Perfect EMA PP V2 first (backtest, chart, alerts for all exec types), then use it as the template for Pack Builder prompts. Swing 1-2-3 is the first Pack Builder test. Legacy packs stay labeled but functional until rebuilt through the scalable pipeline.
- **Legacy packs use generic engine wrapping** — Legacy pack interpreters only detect bar-close triggers. The engine wraps them with LC/CC confirmation logic at the position state machine level. V2 packs create explicit per-execution-type triggers. Both approaches work, but V2 is more testable and explicit.
- **Stop/Take Profit decoupling** — The "swing" RM pack's take profit side is actually R:R math, not swing logic. Split into independent packs: "Swing" (stop-only) + "Risk:Reward" (universal take profit). Each take profit pack should work with any stop loss pack.
