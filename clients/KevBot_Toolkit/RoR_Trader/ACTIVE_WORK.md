# RoR Trader — Active Work Tracker

**Last Updated:** 2026-04-01
**Current Phase:** Phase 2 (Strategy Detail Page Refinements) — IN PROGRESS
**Current Focus:** Client-side date range filtering deployed, QA in progress

---

## QA Troubleshooting Guide

When hitting errors during QA, please include the following in your report:

**Browser console errors:** Copy the full error text (right-click → Copy in Chrome DevTools).

**Railway API logs:** Go to Railway → API service → Logs tab, and filter for these terms:
- **`statement timeout`** — Database query took too long (transient, try refreshing)
- **`Error`** or **`Traceback`** — Python exception with stack trace
- **`ENRICH`** — Our strategy enrichment logging (shows data quality issues)
- **`DETAIL`** — Strategy detail endpoint logging
- **`KPIs`** — KPI computation logging

Grab the last 2-3 minutes of matching logs and paste them in.

**Common issues:**
- **503 (Service Unavailable)** — Usually a Supabase database timeout. Try refreshing. If persistent, check API logs for `statement timeout`.
- **500 (Internal Server Error)** — Code error. Check API logs for `Traceback`.
- **CORS error** — Follows a 500/503. Fix the underlying API error and CORS resolves.
- **"Cannot access X before initialization"** — Terser TDZ error. Variables must be declared before use in source order.

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
- ✅ 2g. **Date range filtering** — Client-side filtering for instant response. Primary KPIs + equity curve update. Extended KPIs fixed (missing dependency in useMemo).
- ✅ 2h. **Brackets setting on Strategy Detail** — Respects display settings.
- ✅ 2i. **Exit trigger display** — Bar Count Exit shows as "[C] Bar Count Exit (Default) > N bars" in summary bar and Configuration tab.
- ✅ 2j. **Trigger display names** — Entry shows "Pack (Variation) > Trigger" format.
- ✅ 2p. **Diagnostic logging** — Frontend console.warn + backend [ENRICH] prefix logging.
- ✅ 2r. **Date range performance** — Client-side filtering from cached stored_trades. Switching ranges is instant.

**Remaining Phase 2 — Proposed Batch Order:**

**Batch A: Equity Curve Polish** (2k + 2m + 2n) — ✅ COMPLETE
- ✅ 2k. **Forward test start date indicator** — Segment color change marks the BT→FWD boundary. Computed from forward_test_start date.
- ✅ 2m. **3-segment equity curve** — BT (blue) / FWD (orange) segments with seamless overlap. Equity curve loads instantly from stored_trades (no Polygon dependency). Per Day mode fills non-trading weekdays.
- ✅ 2n. **Equity curve toggles** — X-axis toggle (Per Trade / Per Day) on chart card. Edge Check (21-MA) wired. HWM toggle has minor rendering issue (deferred). Confidence Bands deferred.

**Batch A Follow-up: Performance vs Plan** (2v)
- ✅ 2v. **Performance vs Plan chart** — Bands centered on plan line, two distinct SD bands, summary KPIs, status badge. Positioned below R-Distribution. Loads instantly from stored_trades.
- 🔍 2w. **PvP: FWD vs Alert toggle** — Add toggle on Performance vs Plan chart to switch the "actual" line between forward test trades and alert trades. Plan line stays the same. Gives transparency on execution quality vs theoretical forward test.
- 📋 2z. **Per-trade alert matching on equity curve** — In Per Trade mode, the green alert line currently uses its own trade numbering. Consider matching each alert trade to its corresponding algo trade number so they line up on the x-axis. Missed trades = gap in alert line, phantom trades = gap in FWD line. Needs investigation — may use Chart & Trades tab matching logic. Review at end of active work phases.
- ✅ 2x. **Sigma badge recalculation** — Backend `compute_sigma_deviation` updated to PvP cumulative formula. Frontend client-side sigma matches. Strategy Detail badges now consistent with PvP chart.
- 📋 2aa. **My Strategies page: alert line + alert sigma** — The MiniEquityCurve on strategy cards doesn't show the green alert overlay. Alert sigma badges show 0.0σ for strategies where `live_executions` isn't populated. Wire alert data through to the cards and investigate why `live_executions` is empty on some strategies.
- 📋 2y. **Forward test: button-only loading** — Forward test computation now only runs on "Update Forward Tests" button click. "Update All Data" does full refresh. Page loads instantly with stored data. Consider adding similar bulk update buttons on My Strategies page.

**Batch B: Pack Display + Naming** (2o + 2s + 2t + 2u)
- 📋 2o. **Stop/Target exec type badges** — Add exec type to stop/target display (prep for candle-close variants in Phase 5)
- 📋 2s. **Stop/Target display names with pack origin** — "Pack (Variation) > Method" format
- 📋 2t. **Bar Count Exit as confluence pack** — Proper pack structure with variations
- 📋 2u. **Trigger naming convention** — Long name vs short name standard, possible display setting

**Batch C: Date Range Enhancements** (2l + 2q)
- 📋 2l. **Custom date range** — Add "Custom" option with start/end date pickers
- 📋 2q. **Show specific date range text** — Thin text below dropdown showing actual dates

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
