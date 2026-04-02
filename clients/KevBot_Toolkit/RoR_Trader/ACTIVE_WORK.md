# RoR Trader — Active Work Tracker

**Last Updated:** 2026-04-01
**Current Phase:** Phase 3 (Hi-Fi Execution & Hold Times) — IN PROGRESS
**Current Focus:** Batch 3 (trade drill-down modal) QA passed. Next: Batch 3b (analyze Hi-Fi), then Batch 4 (PB/CB)

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

### Phase 2: Strategy Detail Page Refinements — COMPLETE

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
- ✅ 2o. **Stop/Target exec type badges** — [L] badge before stop and target in summary bar and Configuration tab. Target shows no badge when "Signal exit only".
- ✅ 2s. **Stop/Target display names** — Client-side formatStopDisplay/formatTargetDisplay show readable strings (e.g., "ATR x1.5", "Swing (5 bars, $0.05 pad)", "2R") instead of raw method names.
- 📋 2t. **Bar Count Exit wiring** — Deferred to Phase 5. Display workaround already in place.
- ✅ 2u. **Trigger naming convention** — Documented. Long = "Pack (Variation) > Trigger", Short = "Trigger".

**Batch C: Date Range Enhancements** (2l + 2q)
- ✅ 2l. **Custom date range** — "Custom" option in dropdown with native date pickers. Client-side filtering. Pattern from PortfoliosPage.
- ✅ 2q. **Show specific date range text** — Thin text showing "Nov 20, 2025 — Mar 31, 2026" inline with the dropdown.

### Phase 3: Hi-Fi Execution & Hold Times — IN PROGRESS

**Status Key:** ✅ QA Passed | 🔍 QA Reviewing | 🔧 Fixing | 📋 TODO

**Batch 1: Foundation**
- ✅ 3a. **1-second bar fetching + caching** — `fetch_1s_bars_for_window()` already existed in data_loader.py. Fixed `_polygon_ticker` → `_to_polygon_ticker` typo. Fixed timezone handling for tz-aware Timestamps.
- ✅ 3b. **Hold time computation** — Engine already computes `bars_held` and `hold_time_seconds` in `get_trade_record()`. Fields present in stored_trades.
- ✅ 3c. **Surface hold times in UI** — Hold column in Algo History + Alert History tables. Avg Hold / Median Hold in Extended KPIs. Alert hold times computed from timestamps. Algo hold times show "--" for older strategies (need full backtest rerun to populate hold_time_seconds). New strategies will have hold times automatically.

**Batch 2: Engine Resolution (1-second precision)**
- ✅ 3d. **Wire 1-second resolution into engine** — `_hifi_resolve_trades()` in backtest_service.py. Resolves every entry/exit bar (no ambiguous-only detection). First-hit-wins for stop/target. Fetches 1-second data from Polygon with day-level caching. Verified: Standard vs Hi-Fi produce different KPIs (Total R: +4.1 vs -14.6 on same 737 trades).
- ✅ 3e. **Strategy Builder fidelity setting** — Standard / High Fidelity toggle. Passes `hifi_mode` to both Load Data and Analyze requests.

**Batch 3: Visualization (trade drill-down modal)**
- ✅ 3f. **Trade zoom API endpoint** — `/trade-zoom` returning 1-second OHLCV bars + trade details. Polygon day-level caching via `fetch_1s_bars_for_window`.
- ✅ 3g. **Trade drill-down modal** — Click entry/exit time in Algo History → TradeZoomModal opens with 1-second candles via SyncedChartPane. Entry/exit arrow markers, trade details card. Stop/target/entry price lines (price lines need SyncedChartPane support refinement). Stepped indicators + confluence heatmap deferred to refinement pass.

**Batch 3 Refinements:**
- ✅ 3f-r1. **Stepped indicators on drill-down modal** — Original-timeframe indicators shown as stepped horizontal lines. Filtered to strategy-relevant indicators only.
- ✅ 3f-r2. **Entry/exit + and × markers** — Algo fill shown as + (cross shape, green) at exact fill price. Alert fill shown as × (xcross shape, orange). Uses patched lightweight-charts shapes that can overlap/intersect to form a star when prices match (zero slippage). Snapped to nearest 1-second bar. C-type timestamp shift applied.
- ✅ 3f-r3. **Alert price data wiring** — Alert entry/exit prices now captured in algoMatches and passed through to drill-down modal.
- ✅ 3f-r4. **Candlestick colors from display settings** — Drill-down modal reads candle colors from useChartPrefs(), matching main chart behavior.
- 📋 3f-r5. **Confluence heatmap on drill-down** — Deferred until Batch 4 (PB/CB fidelity).

**Batch 3b: Analyze Tabs Hi-Fi (must complete before leaving Phase 3)**
- ✅ 3h. **Wire Hi-Fi into analyze endpoints** — All 4 analyze implementations (entry, exit, stop, target) now call `_hifi_resolve_trades()` when hifi_mode=True. TF Conditions/General modes use the base backtest's already-resolved trades.

**Batch 4: Polish**
- 📋 3i. **PB/CB fidelity variants on TF Conditions** — Like exec types on triggers. Each condition generates [PB] and [CB] variants as separate selectable items. Standard fidelity hides CB. HiFi shows both.
- 📋 3j. **Mass Builder fidelity setting** — HiFi option in Mass Builder config.

### Phase 4: Replay + Pack Validation — NOT STARTED

Replay is critical for validating confluence packs before they're locked. Moved from original Phase 4 scope (dedicated chart test page — now covered by Strategy Builder + drill-down modal).

- [ ] 4a. **Replay engine** — Bar-by-bar playback simulating real-time execution. Shows indicators updating, triggers firing, positions opening/closing in sequence. Essential for verifying confluence packs behave correctly before finalizing.
- [ ] 4b. **Replay UI** — Play/pause/step controls, speed selector (1x/2x/5x/10x), bar-by-bar stepping. Can be integrated into Strategy Builder or as a dedicated panel.
- [ ] 4c. **Pack validation environment** — Replay + trigger event log showing exactly when and why each trigger fired. Detects flicker logic, painting issues, and execution timing problems. Used during Pack Builder workflow before a pack is saved and locked.
- [ ] 4d. **SSE progress tracking** — Server-Sent Events for real scenario counts during backtests and replay.

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

## Post-Phase Polish (After Active Work Phases Complete)

These items improve consistency and UX but are NOT on the critical path to trading.
Tackle after Phases 1-6 are complete and strategies are actively placing trades.

- 📋 **My Strategies: replace MiniEquityCurve with shared EquityCurve** — The mini equity curve on strategy cards is a separate inline component that doesn't share display settings, alert overlay, or segment coloring. Replace with the shared EquityCurve component in mini mode for consistency.
- 📋 **My Strategies: alert sigma from alerts API** — Alert sigma currently reads from `live_executions` (empty for many strategies). Compute client-side from alerts API like Strategy Detail does, or backfill `live_executions` from alert history.
- 📋 **My Strategies: alert equity overlay on cards** — Wire green alert line onto strategy card equity curves.
- 📋 **My Strategies: bulk "Update Forward Tests" button** — Like Strategy Detail's button but for all strategies at once.
- 📋 **Per-trade alert matching on equity curve** — Match alert trades to algo trade numbers for aligned x-axis comparison. Detect missed/phantom trades visually. (Item 2z)
- 📋 **Gradient fill toggle** — Settings page saves the preference but it's not consistently applied to all equity curves.
- 📋 **Edge Check refinement** — Current implementation is simple 21-MA. Streamlit uses 21-MA + Bollinger Bands (2SD) as a shaded region. Port the full pattern from `_add_edge_check_traces()` in app.py.
- 📋 **Confidence Bands** — Needs statistical computation (Monte Carlo or parametric). Currently checkbox exists but not implemented.

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
