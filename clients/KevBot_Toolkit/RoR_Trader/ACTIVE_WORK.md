# RoR Trader — Active Work Tracker

**Last Updated:** 2026-04-01
**Current Phase:** Phase 2 (Strategy Detail Page Refinements) — IN PROGRESS
**Current Focus:** QA round 3 — fixing date range, brackets, trigger names, exec badges on detail page

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

**Status Key:** ✅ QA Passed | 🔍 QA Reviewing | 🔧 Fixing | 📋 TODO

**Priority items (from Strategy Builder triage):**
- ✅ 2a-new. **Stop/Take Profit pack restructuring** — Stop packs show only stop params, target packs show only target params. Strategy Builder uses dedicated hooks. "Swing" is stop-only, new "Risk:Reward Target" template added (take-profit-only, works with any stop).
- ✅ 2b-new. **Strategy Builder defaults (entry, exit, stop)** — Zustand store with localStorage. Auto-populates on mount. "Save as Default" / "Clear Defaults" buttons.

**Strategy Builder items (QA passed):**
- ✅ 2c. **Trigger names with pack origin** — "Group > Trigger" format. Strategy Builder groups by pack name.
- ✅ 2d-sb. **TF Conditions: fidelity badges** — TF Conditions tab shows [PB], General tab shows no badges.
- ✅ 2e-sb. **Display settings wiring (Strategy Builder)** — ExecBadge and FidelityBadge respect color, shape, and brackets from useDisplayStore.

**Display Settings items:**
- ✅ 2f-ds. **Display settings persistence** — Color, shape, brackets persist across navigation. Zustand store syncs on save and hydration.

**Strategy Detail items:**
- 🔧 2g. **Date range: Extended KPIs + equity curve not updating** — Primary KPIs update correctly. Extended KPIs and equity curve use separate data sources that bypass the filtered trades. Need to derive these from the same filtered dataset.
- ✅ 2h. **Brackets setting on Strategy Detail** — Now respects display settings.
- 🔧 2i. **Exit trigger showing blank** — "Signal exit only" is confusing. Bar Count Exit (4 bars) should appear as an exit trigger in the summary bar with pack naming convention: "Bar Count Exit (Default) > 4 Bars". Exit triggers should follow same naming as entry triggers.
- ✅ 2j. **Trigger display names on Configuration tab** — Entry shows "Pack (Variation) > Trigger". Working for entry and configuration view.
- 📋 2k. **Forward test start date indicator** — Equity curve should show when FWD started (date label or color boundary). Not currently visible.
- 📋 2l. **Custom date range option** — Add "Custom" to date range dropdown with start/end date pickers.
- 📋 2m. **3-segment equity curve** — BT (blue) → FWD (orange) → Alerts (green) coloring per display settings. Currently single color.
- 📋 2n. **Equity curve toggles** — HWM, Edge Check, Confidence Bands toggles currently non-functional on Strategy Detail.
- 📋 2o. **Stop/Target exec type badges (prep for Phase 5)** — Stops and targets should show exec type so we can later add candle-close variants for TradingView parity testing.
- ✅ 2p. **Diagnostic logging** — Frontend console.warn + backend [ENRICH] prefix logging deployed and working.
- 📋 2q. **Date range: show specific dates** — Display the actual date range in thin text below the dropdown (e.g., "Mar 1 - Mar 31, 2026") so the user knows exactly what window they're looking at.
- 📋 2r. **Date range: performance optimization** — Cache at least the strategy default period of data. Date ranges within the cached window should filter client-side for instant response. Only fetch new data if the range extends beyond what's cached. Consider scaling implications for 10,000+ strategies.
- 📋 2s. **Stop/Target display names with pack origin** — Show "Pack (Variation) > Method" for stop and target in summary bar, same as entry triggers. May need display setting for name verbosity (long vs short).
- 📋 2t. **Bar Count Exit as confluence pack** — Treat Bar Count Exit structurally like other confluence packs with a default pack and variations for different bar counts. Should appear in summary bar as an exit variable.
- 📋 2u. **Trigger naming convention** — Establish long name (Pack (Variation) > Trigger) and short name (Trigger only). Long name used in Configuration and summary bar. Short name available for compact displays. Could be a display setting for user preference.

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

**Priority items (foundational pack UX):**
- [ ] 5a-new. **Pack variation nesting** — All pack pages (TF Confluence, General, Stop Loss, Take Profit) should group variations under their parent template with expand/collapse chevron. First variation = "Default", user-created variations nest underneath. Per Frontend Migration Plan: "Nested under defaults — Variations indent under parent template with expand/collapse chevron inside the card."
- [ ] 5b-new. **Legacy pack cleanup** — Mark all current TF Confluence packs (except EMA Price Position V2) as "Legacy (Default)" in their display names. Keep them fully functional — legacy packs use the engine's generic LC/CC wrapping which works but isn't purpose-built like V2. Legacy label stays until packs are rebuilt through Pack Builder or manually updated to V2's explicit trigger variant structure. Do NOT delete or disable legacy packs.

**Pack Builder pipeline (get EMA PP V2 perfect → inform Pack Builder → scale):**
- [ ] 5c. Extend pack_spec.py validation for exec_variants
- [ ] 5d. Update pack_builder_context.md with exec_variants docs + examples (modeled on EMA PP V2 as the reference)
- [ ] 5e. Update pack_builder.py LLM prompt (include flicker logic, painting logic, execution fidelity checks)
- [ ] 5f. Update pack_registry.py to register LC/CC triggers
- [ ] 5g. **Swing 1-2-3 as first Pack Builder test case** — Don't fix the current broken swing_123 manually. Instead, use it as the first real test of the Pack Builder pipeline. Validates that the system can scalably produce packs with proper CC execution, correct trigger detection, and accurate backtests.
- [ ] 5h. Backward compat: exec_variants optional, existing packs default C-only
- [ ] 5i-new. **Trade qualification filter in Strategy Builder** — Add TQ filter dropdown to the filter/sort modal. Requires backend TQ rule application logic to be built first. Design exists in Frontend Migration Plan (Portfolio Requirements V5 spec).

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
