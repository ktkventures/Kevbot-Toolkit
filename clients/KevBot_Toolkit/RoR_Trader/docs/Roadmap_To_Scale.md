# RoR Trader — Roadmap to Scale

**Created:** 2026-04-05
**Goal:** Get the core building blocks fully correct so that adding packs, strategies, and portfolios at scale is reliable and consistent. Then enable AI agents to create and test thousands of strategies autonomously.

**Guiding Principles:**
- One engine path for all trade generation (unified engine)
- Modular execution types (testable in isolation, new types without engine changes)
- Pack Builder produces reliable, consistent packs without manual intervention
- Alerts and webhooks fire at exactly the same times as backtest signals
- Scale to thousands of strategies, portfolios, and users

---

## How We Work Through Each Milestone

**1. Plan First**
Before starting any milestone, enter plan mode. Thoroughly explore the relevant code, understand the current state, identify risks, and design the approach. Present the plan with clear steps and exit criteria for approval before writing any code.

**2. Self-Test, Then Hand Off for Verification**
Claude iterates and self-tests locally (API endpoints via curl, frontend compilation checks, data flow verification). Once a batch of changes is working, hand off specific things for Kevin to verify visually in the browser — charts rendering correctly, UX feeling right, data looking accurate. This serves two purposes: Kevin stays familiar with what changed, and visual QA catches things automated checks miss.

**3. Incremental Commits**
Commit and push working batches frequently. Don't accumulate a massive diff. Each commit should leave the system in a working state. If something breaks, it's easy to identify which commit caused it.

**4. No Mock Data in Finished Features**
Placeholder/mock data is fine during development, but before a milestone is marked complete, all features must use real API data. No `{{template}}` strings, no hardcoded empty arrays, no setTimeout placeholders in shipped code.

**5. Consistency Is the Priority**
If a feature works in the Strategy Builder, it must work the same way in the Sandbox, in Strategy Detail, in Mass Builder, and in live alerts. One code path, one engine, one set of results. If we find a bug in one place, check all the others.

**6. Build Bottom-Up**
The stack builds in order: Packs → Strategies → Mass Builder → Portfolios → Scale. Each layer must be solid before building the next. Don't rush to portfolios if strategies aren't reliable.

**7. Task Classification**
Each task within a milestone is tagged with one of three priorities:
- **[Required]** — Must be done before moving to the next milestone. Would cause problems downstream if skipped.
- **[Polish]** — Should be done during this milestone if time allows. Cosmetic, UX, or minor behavior tweaks. Can be deferred without causing downstream issues.
- **[Deferred]** — Noted for later. Good idea, but not blocking anything. Come back when the area is being polished.

At milestone completion: all Required items done, Polish items reviewed (do now or defer), Deferred items stay in the doc. Kevin's feedback during a milestone gets added with the appropriate tag.

---

## Current State (as of 2026-04-05)

### What's Working
- **Pack Builder AI Integration** — Batches 1-5 complete. AI generates packs (Claude/OpenAI), validates, installs, Sandbox tab for testing
- **User Packs Page** — Lists installed packs, detail view with Sandbox, real code display, AI Fix capability
- **Strategy Builder** — Full backtest + analysis pipeline for built-in packs
- **Strategy Detail** — Live data, forward testing, equity curves, alert history, trade drill-down
- **Dashboard** — Aggregated KPIs, equity curves, strategy health
- **Portfolios** — Multi-strategy portfolios with Monte Carlo, buying power tracking
- **Data Provider** — Polygon.io REST + WebSocket (stocks), 1-second bars for Hi-Fi
- **Auth** — Supabase multi-user with JWT, Railway deployment

### What Needs Fixing
- **Engine parity** — User packs use batch `generate_trades()` fallback instead of unified engine
- **Execution type hardcoding** — C/L/LC/CC logic is hardcoded in PositionStateMachine
- **Webhook pipeline** — Not wired for the Next.js frontend (Streamlit-era design)
- **Ralph engine (live alerts)** — Works but isn't connected to the new frontend
- **ACTIVE_WORK.md** — Outdated, refers to the old 6-phase plan from earlier sessions

---

## The Path Forward

### Milestone 1: Engine Parity (Option A) — COMPLETE
**Completed:** 2026-04-05

All packs (built-in and user) now flow through the unified engine. The `generate_trades()` batch fallback has been removed. User pack interpreter states and trigger booleans are read from pre-computed DataFrame columns and merged into the unified engine's bar-by-bar loop.

**What was done:**
- [x] 1a-b. Unified engine reads pre-computed user pack columns (interpreters + triggers) via `user_pack_data` parameter in `process_bar()`
- [x] 1c. Removed `_use_batch` / `generate_trades()` fallback from `backtest_service.py`
- [x] 1d. Bar count exit detection: Sandbox auto-passes `bar_count_exit=4` when bar count trigger selected
- [x] 1e. Verified: RSI Zones produces trades via unified engine
- [x] 1f. Verified: EMA PP V2 (built-in) has no regression
- [x] Added `resolve_strategy_requirements()` user pack support via pack_registry query
- [x] Added confluence dropdown (multi-select from enabled groups) and auto-derived secondary TFs
- [x] Fixed SyncedChartPane oscillator alignment (time-based sync)

---

### Milestone 2: Execution Type Extraction
**Priority:** High — enables modular execution types, testable in isolation
**Effort:** 1-2 weeks

Extract the 4 execution type branches from `PositionStateMachine` into pluggable modules with a workflow schema. Build an Execution Types management page with vertical step visualization. The workflow schema is designed to translate cleanly into a full drip-campaign-style visual builder in the future — no data migration or backtest invalidation required.

**Design Decisions:**
- Each execution type is defined as a **workflow** (ordered list of steps) + **parameters**, stored as JSON
- Steps include: plot_marker, fire_webhook, set_position, wait, conditional branch
- The 4 existing types are extracted as the first workflow definitions — behavior stays identical
- Variations = same type with different parameter values (e.g., LC with bail_at_market vs bail_at_limit)
- The engine executes workflows the same way it executes the current hardcoded branches — trade records are identical
- **Future upgrade path:** Visual drip-campaign builder adds/removes/reorders steps. Core fill logic unchanged. Data stays valid.
- **Data integrity:** Execution types define HOW a fill happens (price, timing). Future workflow steps add observable actions AROUND the fill (webhooks, markers, waits) but don't change the fill itself. Existing backtest/forward test/alert data remains valid.

**Tasks:**
- [ ] 2a. [Required] Define `ExecutionType` interface and workflow step schema (JSON format)
- [ ] 2b. [Required] Extract Bar Close (C) execution logic into module with workflow definition
- [ ] 2c. [Required] Extract Level (L) execution logic into module with workflow definition
- [ ] 2d. [Required] Extract Level-Close (LC) execution logic into module with workflow definition
- [ ] 2e. [Required] Extract Close-Close (CC) execution logic into module with workflow definition
- [ ] 2f. [Required] Refactor `PositionStateMachine` to call execution type modules instead of switch/if branches
- [ ] 2g. [Required] Create execution type registry (similar to pack_registry) — load, register, list
- [ ] 2h. [Required] Build Execution Types page — vertical step list view, parameter editing, variations
- [ ] 2i. [Required] Verify: existing strategies produce identical trades after refactor (no regression)
- [ ] 2j. [Polish] Isolation testing UI: synthetic signal → watch execution type behavior
- [ ] 2k. [Deferred] Update Ralph engine to use the same execution type modules (parity with unified engine)
- [ ] 2l. [Deferred] Visual drip-campaign workflow builder (drag-and-drop steps, conditional branches)
- [ ] 2m. [Deferred] **Split Confirmed into LC and CC as separate modules** — Currently LC and CC are grouped in one ConfirmedExecution module because they share confirmation/bail logic. Future: split into separate modules (4 cards on the page), each with its own reference_bar, order_type, confirm_bar_offset, bail_action parameters. Move these params from TF Confluence pack's _exec_config to the execution type definition itself.
- [ ] 2n. [Deferred] **Parameters on execution type, not confluence pack** — Currently execution type params (reference_bar, order_type, etc.) are stored on the confluence group's _exec_config and inherited. Future: store on the execution type variation directly. Confluence pack just selects which execution type variations to enable.
- [ ] 2o. [Deferred — Future Core Type] **Oscillator Cross (OC) execution type** — A new core execution type for intra-bar entries on oscillator-based triggers (Stochastic K/D crossover, MACD line/signal crossover, RSI midline cross, etc.). Currently L-type only works for triggers where one PRICE crosses another PRICE (e.g., bar high crosses VWAP). Oscillators have a single value per bar calculated from bar OHLC, so true intra-bar crosses can't be detected without recomputing the indicator at sub-bar resolution. The OC execution type would leverage Hi-Fi mode infrastructure to recompute the oscillator on 1-second bars for the entry window, detect the precise cross moment, and record the entry at that timestamp. This is a new CORE execution type (not a variant of L), since the underlying mechanism is fundamentally different (sub-bar indicator recomputation rather than cached level cross). Examples: MACD line crosses signal line, Stochastic K crosses D, RSI crosses 50, CCI crosses zero. Many traders rely on these as the primary entry trigger but currently the system can only fire them at bar close, missing the actual cross moment by up to a full bar duration. **Architecture note:** Just like C and L are core execution types, OC would be a new core type. C and L already serve as the foundation for most other execution types — OC would extend that foundation to oscillator-aware events.

**Completed:** 2026-04-06. All Required tasks done. Parity verified (EMA PP V2 = 11 trades/-10.0R, RSI Zones = 13/-14.7R — identical before and after).

**Exit Criteria:** Execution types are pluggable modules with workflow schema. The 4 existing types produce identical trades to the pre-extraction engine. The Execution Types page shows each type with its step sequence and parameters. Users can create variations with different parameter values.

---

### Milestone 3: Webhook Pipeline for Next.js — COMPLETE
**Completed:** 2026-04-06

Frontend wired to existing backend alert/webhook infrastructure. Monitor start/stop, template CRUD, test delivery, exec_type on alerts all working.

**What was done:**
- [x] 3a. Monitor start/stop buttons on AlertsPage engine status strip
- [x] 3b. Webhook template list, create, detail, test wired to API
- [x] 3c. Backend pipeline already handles alert → webhook delivery (verified via test endpoint)
- [x] 3d. exec_type added to alert records + {{exec_type}} webhook placeholder
- [x] 3e. End-to-end tested: monitor start/stop, template list, webhook test delivery to httpbin.org
- [ ] 3f. [Polish] Account-level webhook templates (deferred)

**Known issue:** Template create has minor DB schema mismatch (exchange column doesn't exist). Default templates already in DB from Streamlit era. Fix when template editing is needed.

**Full live testing** (strategy → alert → webhook) deferred to Railway deployment after Milestones 4-7 are complete.

---

### Milestone 3.5: Execution Type Architecture Finalization — COMPLETE
**Completed:** 2026-04-06

Split execution types into 4 separate modules with clean parameter schemas. Execution Types page with card/detail pattern, enable/disable toggles, parameter editing. Execution Defaults removed from Pack Builder and TF Confluence packs.

**What was done:**
- [x] 3.5a. Split ConfirmedExecution into LevelCloseExecution (LC) + CloseCloseExecution (CC)
- [x] 3.5b. Cleaned parameters: removed non-functional reference_bar from all types, removed fixed CC params
- [x] 3.5c. Enable/disable toggle with per-user persistence (stored in user_settings)
- [x] 3.5d. ExecutionTypesPage: single-column cards with enable toggle + click-to-detail
- [x] 3.5e. Detail view: Parameters tab (editable) + Workflow Steps tab
- [x] 3.5f. TF Confluence pack exec config: not yet removed from TfConfluencePage (deferred — legacy packs still reference it)
- [x] 3.5g. Removed Execution Defaults card from Pack Builder Step 3
- [ ] 3.5h. [Deferred] Add test URL input to webhook template detail
- [ ] 3.5i. [Deferred] Variation support (nested under parent)

**Final parameter schemas:**
- C: order_type
- L: order_type, hold_seconds
- LC: order_type, confirm_bar_offset, bail_action, hold_seconds
- CC: no parameters (fixed behavior)

---

### Milestone 4: Execution Type Polish
**Priority:** High — execution types must be validated before building packs on top of them
**Effort:** 3-5 days

Flesh out the execution type detail page with validation/testing capabilities. Define how execution types should behave going forward (not just documenting legacy behavior). The goal: each execution type can be tested in isolation, proving it works correctly for ANY trigger before it's applied to confluence packs.

**Tasks:**
- [ ] 4a. [Required] **Simulation tab** — On execution type detail page. Pick a symbol + timeframe, system generates a synthetic trigger at a specific bar, shows: fill price, confirmation timing, bail behavior, stop/target placement. Mini chart with entry/exit markers.
- [ ] 4b. [Required] **Webhook step integration** — Add webhook event steps to workflow definitions. Each step shows which webhook event would fire (entry_signal, confirm, bail, etc.). This defines the execution PROCESS, not just the engine behavior.
- [ ] 4c. [Required] **Execution type variations** — Create variations with different parameter values (nested under parent). Default is immutable. Variations are separate entries with their own names.
- [ ] 4d. [Required] **Execution type immutability** — Default execution types cannot be edited. Users must create a variation to change parameters. Protects forward test data integrity.
- [ ] 4e. [Required] **Wire execution types into trigger generation** — Enabled execution types × pack triggers = all trigger options in Strategy Builder. Remove dependency on `_exec_config` in confluence groups.
- [ ] 4f. [Polish] **Execution type documentation** — Each type has a clear description of when it fills, at what price, what confirmation/bail behavior applies, and what webhook events fire.
- [ ] 4g. [Deferred] **Custom execution type creation** — Users can define new execution types via the drip-campaign workflow builder.

**Completed:** 2026-04-06. Phases 1-4 done. Webhook steps, simulation tab, variations, trigger filtering all working.

---

### Milestone 4.5: Execution Type Creation & Validation — COMPLETE
**Completed:** 2026-04-08

Scenario-based educational reference system with replay, custom scenario generator, and validation-in-Strategy-Builder architecture.

**What was done:**
- [x] 4.5c. Scenario reference system — 13 static scenarios across 5 categories with replay mode, workflow traces, Hi-Fi 1-second drill-downs
- [x] 4.5g. Replay mode — Second-by-second replay with forming candle, workflow states, Hi-Fi charts
- [x] 4.5c2. Expanded scenarios + category tabs — Signal exit, L-Type, LC/CC confirmation/bail, gap edge cases
- [x] 4.5c3. Custom scenario generator — Own tab with trigger/pack selectors, runs real backtest, cherry-picks trades
- [x] 4.5h. Removed Sandbox tabs from Execution Types and User Packs detail views
- [x] 4.5i. "Test in Strategy Builder" button on Execution Types and User Packs detail pages
- [x] 4.5j. Reframed Scenarios as educational — descriptions clarify conceptual illustrations, link to Strategy Builder for validation

**Key Decision (2026-04-08): Validation belongs in Strategy Builder, not module pages.**
- Scenarios tab is EDUCATIONAL — explains concepts, not validation
- Sandbox tabs removed from Execution Types and User Packs pages
- Strategy Builder is the single source of truth for validation
- "Test in Strategy Builder" shortcut button on module detail pages

**Also completed during M4.5 session:**
- Enhanced Trade History table in Strategy Builder — clickable trade #, entry/exit times, exec badge
- Trade Replay Modal — full scenario replay (main chart + 2 hi-fi + replay controls) from trade # click
- Trade Workflow Modal — execution workflow steps from exec badge click
- Modal portal fix — all modals render above sidebar via createPortal
- Trade drill-down indicator fix — 1s charts only show entry/exit trigger indicators, not all confluence
- Heatmap fix — empty confluence = no heatmap (removed show-all fallback)
- Exec type badges on stop loss / take profit packs (Strategy Builder selectors + optimizable variables)
- Backend refactors: `build_single_trade_scenario()`, `classify_and_serialize_chart_data()` helpers

**Deferred to future milestones:**
- 4.5a. Unique 1-5 char badges per execution type variation
- 4.5b. Mini chart on simulation trace
- 4.5d. AI-assisted execution type creation
- 4.5e. Execution type builder page
- 4.5f. Code tab for generated execution types
- Hi-Fi cross (+) marker not rendering during replay mid-step
- CB timeline computation performance (loads 30 days of 1-min bars for all groups)

---

### Milestone 5: Pack Builder & User Packs Polish
**Priority:** Medium — reliability of pack creation
**Effort:** 1-2 weeks

Ensure packs created through the Pack Builder are consistently reliable and work seamlessly across all features. Now that execution types are separate pluggable modules and validation happens in Strategy Builder, packs can be built and tested with confidence.

**Important context:** All existing TF Confluence packs (including EMA PP V2) were built under the old architecture where execution types were configured via `_exec_config` on the confluence group. These packs still work, but new packs should be built using the current execution type module system. Legacy cleanup happens first so there's a clear line between old and new.

**Completed Tasks:**
- [x] 5a. **Legacy pack cleanup** — Orange "Legacy" badge on all built-in TF Confluence packs + User Packs cards. Info banner on TF Confluence page. Derived from `is_default` field (not persisted as tag).
- [x] 5c. **Updated pack_builder_context.md** — Execution types documented as separate modules. `_exec_config` marked as legacy. Verification workflow section added. Reserved names updated with RSI prefix/columns. Trigger neutrality guidance added.
- [x] 5e. **Chart Preview tab** — Backend endpoint `POST /api/packs/builder/user-packs/{slug}/preview`. State-colored candles, overlay indicator lines, oscillator pane (separate chart), trigger markers, state legend. State/trigger tables wired to real data.
- [x] 5j. **Signal Validation tab** — Runs pack on configurable data (symbol/timeframe/30-180 days). Summary metrics (total signals, avg bars between, state coverage, all states reached). State distribution bar. Per-trigger breakdown with frequency badges.
- [x] 5l. **Trigger direction/type neutrality** — All triggers now direction/type agnostic (`BOTH`/`BOTH`). Backend returns all triggers regardless of type. Pack spec accepts `BOTH` as valid type. Built-in templates updated. Frontend uses single trigger fetch.

**Remaining Tasks:**
- [ ] 5b. [Required] Fix remaining validation gaps (discovered during testing)
- [ ] 5d. [Required] Add Swing 1-2-3 as Pack Builder test case — validate CC triggers work end-to-end
- [ ] 5f. [Required] Create several user packs through the full Pack Builder flow and verify each one end-to-end in Strategy Builder (run backtest → inspect Enhanced Trade History → replay trades)
- [ ] 5g. [Polish] Pack status workflow: Verification → Private → Public (persist status, gate Strategy Builder visibility)
- [ ] 5h. [Polish] Pack versioning: installed pack is immutable, new versions create new packs
- [ ] 5i. [Polish] Delete protection: warn if strategies reference the pack
- [ ] 5k. [Polish] **Bar count exit variations** — Read N from confluence group parameters
- [ ] 5m. [Deferred] **Unify User Packs into TF Confluence page** — Filter by author instead of separate page
- [ ] 5n. [Deferred] **Parity Simulator tab** — Requires Ralph engine replay capability or sufficient live alert history. The Strategy Detail "Alert Trades" tab already compares backtest vs alert timing per-strategy. Parity Simulator would extend this to per-pack diagnostics. Depends on live engine wiring being fully validated.

**Exit Criteria:** Legacy packs clearly labeled. User can create 10 different packs through the Pack Builder and every one works correctly in Strategy Builder and live alerts without manual intervention. Validation flow: Pack Builder → install → Strategy Builder backtest → Enhanced Trade History → Trade Replay.

---

### Milestone 5.5: Stops/Targets Modularization — COMPLETED 2026-04-09
**Priority:** Critical — must happen before building more strategies/packs
**Effort:** 2 days (planning + implementation + verification)
**Status:** Complete and verified end-to-end. Shipped on `dev` branch (backup: `dev-backup-pre-m5.5`).

**Core deliverables:**
- Pluggable registry: `src/stop_target_methods.py` (StopMethod / TargetMethod base classes + STOP_METHODS / TARGET_METHODS dicts)
- Engine refactor: `_compute_stop` / `_compute_target` dispatch through registry; hardcoded if/elif removed
- Exec types L/C/LC/CC implemented in `PositionStateMachine._check_stop_hit` / `_check_target_hit`
- All four call sites in `check_exit` and `check_exit_bar_close` integrated; `check_exit_tick` only fires for L-type
- CC state via `pending_stop_confirm_bar` / `pending_target_confirm_bar` on PositionState
- Frontend: variant list per pack × exec_type (matches entry/exit trigger pattern); selection persists through backtest, save, and analyze paths; supported_exec_types flow from API
- Ralph engine inherits the refactor (uses unified PositionStateMachine)
- Trade record carries `stop_exec_type` / `target_exec_type` for chart marker shift logic

**Bugs found and fixed during verification (4 total):**
1. **`_resolve_configs` exec_type drop** (`api/routers/backtest.py`) — analyze, trade-zoom, trade-replay endpoints were silently dropping `req.stop_exec_type` / `req.target_exec_type`, falling back to L-type regardless of user selection. Fixed: now injects exec_type same as `backtest_service.run_backtest`.
2. **`_hifi_resolve_trades` overwriting C/LC/CC fills** — Hi-Fi resolution was running `_walk_1s_for_exit` (L-type only) for every stop/target, overwriting C/LC/CC bar-close fills with L-type stop-level fills. Fixed: skip Hi-Fi resolution for trades whose stop_exec_type or target_exec_type isn't L.
3. **`_walk_1s_for_exit` not gap-aware** — Even for L-type stops, the Hi-Fi walk recorded `exit_price = stop_price` regardless of whether the 1-second bar opened past the stop. This capped every overnight gap-down loss at exactly -1R. Fixed: now uses `min(stop, bar_open)` for LONG and `max(stop, bar_open)` for SHORT, matching the engine's gap-aware fill logic.
4. **Static "L" badge in Optimizable Variables module** (`StrategyBuilderPage.tsx`) — `OptimizableVariables` was reading `stopPack.execType` (the pack's intrinsic value, always 'L') instead of the user-selected `stopExecType` from React state. Fixed: now reads from selected exec type, updates dynamically.

**Test coverage (all passing):**
- `test_stop_target_parity.py` — 33 cases (old hardcoded vs new registry produce identical values)
- `test_exec_type_hits.py` — 20 cases (L/C/LC/CC dispatch for stop and target)
- `test_unified_parity.py` — 27 cases (L-type behavior fully preserved across full unified engine)
- `test_c_type_swing_no_lookahead.py` — 4 cases (C-type swing stop sanity, no -1R cap)
- `test_c_type_full_path.py` — 4 cases (end-to-end check_exit with synthetic bars)
- `test_c_type_real_backtest.py` — synthetic backtest, R range -2.73 to -1.10 (not capped)
- `test_l_type_gap_down.py` — 4 cases (L-type gap-aware fills, R = -5 and -100 on gaps)
- `test_overnight_gap_full_cycle.py` — drives process_bar through entry → overnight gap → exit (R = -4.24 verified)
- **Total: 90+ test cases passing**

**User-verified end-to-end:** The C-type swing stop produces correct gap losses (e.g. -209R on overnight hold). L-type wick-based stops correctly produce -1R losses on intra-bar wicks. Gap-down losses now correctly exceed -1R for both C-type and L-type Hi-Fi paths.

**Why this is urgent:** Currently stop loss and take profit methods are hardcoded `if/elif` branches in `unified_engine.py` (lines ~1995-2070). Every strategy created today references a method by string (`"atr"`, `"swing"`, etc.) that's interpreted by hardcoded engine logic. If we build out more user packs and strategies before fixing this, we accumulate "legacy stop strategies" that may not behave identically when we later refactor — invalidating their forward test data and creating the same legacy bucket problem we just fixed for execution types.

**The principle:** Stops and targets should follow the same modular architecture as entry/exit triggers. Each method is a pluggable pack with its own logic. Each pack can declare which execution types it supports (C, L, LC, CC). The engine treats them like first-class indicators.

**Architecture design:**

1. **`StopLossPack` base class** (`src/stop_loss_packs.py`) with methods:
   - `compute_initial_stop(entry_price, direction, bars_lookback) -> float`
   - `compute_trail(current_stop, entry_price, current_bar) -> float | None` (optional)
   - `should_move_to_breakeven(entry_price, current_pnl_r) -> bool` (optional)
   - `supports_exec_types() -> list[str]` (returns subset of C/L/LC/CC)

2. **Individual stop modules** (one per method):
   - `ATRStopPack` — uses ATR multiplier
   - `FixedDollarStopPack` — fixed dollar offset
   - `PercentageStopPack` — percentage from entry
   - `SwingStopPack` — N-bar swing low/high
   - `BreakevenStopPack` — initial stop + breakeven trigger at R milestone
   - `ATRTrailingStopPack` — ATR stop that ratchets

3. **`TakeProfitPack` base class** with similar pattern:
   - `compute_target(entry_price, stop_price, direction) -> float | None`
   - `supports_exec_types() -> list[str]`

4. **Individual target modules:**
   - `RiskRewardTargetPack` — N:1 R-multiple
   - `ATRTargetPack` — ATR-based target
   - `FixedDollarTargetPack`
   - `SwingTargetPack`
   - `NoTargetPack` — exits via stop, signal, or bar count only

5. **Execution type integration:**
   - Each stop/target gets an `exec_type` field (default `"L"` for backward compat)
   - Engine respects exec_type when checking stop/target hit:
     - **L**: hit on intra-bar touch (current behavior — `low <= stop` for LONG)
     - **C**: hit only if `close < stop` (close-based, filters wicks)
     - **LC**: touch intra-bar + bar close confirms past stop
     - **CC**: bar closes past stop + next bar also confirms
   - This adds wick filtering for traders who want it without changing existing strategies

6. **Pack registry pattern:**
   - `stop_loss_pack_registry.py` and `take_profit_pack_registry.py` register all available packs
   - Each pack has metadata: `id`, `name`, `category`, `description`, `parameters_schema`
   - The frontend fetches the registry to populate the Strategy Builder dropdowns dynamically
   - New stops/targets added without engine changes — just add a new pack module and register it

7. **Backtest parity verification:**
   - Run a regression test: existing strategies with `method="atr"` and default L exec must produce IDENTICAL trade records before and after the refactor
   - Lock down with stored fixtures comparing `entry_price`, `stop_price`, `target_price`, `r_multiple` for a sample strategy across 90 days

**Tasks:**
- [x] 5.5a. [Required] Define `StopMethod` and `TargetMethod` base classes with method signatures
- [x] 5.5b. [Required] Extract existing stop methods into individual classes
- [x] 5.5c. [Required] Extract existing target methods into individual classes
- [x] 5.5d. [Required] Refactor `unified_engine.py` stop/target logic to use registry dispatch (delete hardcoded if/elif)
- [x] 5.5e. [Required] Build registry pattern + propagate `supported_exec_types` through API serialization
- [x] 5.5f. [Required] Add `exec_type` field to stop/target configs (default `L` for backward compat)
- [x] 5.5g. [Required] Update engine stop/target hit detection to respect exec_type (C/L/LC/CC behaviors)
- [x] 5.5h. [Required] Backtest parity test — existing strategies must produce identical trades
- [x] 5.5i. [Required] Update Strategy Builder UI: exec type dropdown alongside stop/target pack selectors
- [x] 5.5j. [Polish] Show selected exec_type badge on stop/target pack cards in Strategy Builder
- [ ] 5.5k. [Polish] Allow pack authors to create custom stop/target packs via Pack Builder (deferred to M9)

**Exit Criteria:** Stop loss and take profit methods are pluggable modules with no hardcoded engine logic. Each pack can declare its supported execution types. Adding a new stop or target type requires zero engine changes. Existing strategies pass parity test (identical trades before/after refactor). Strategy Builder shows exec type badges and allows users to choose execution timing for stops and targets.

**Related future work (deferred):**
- **Task 2o (Oscillator Cross / OC execution type)** — A new core execution type for intra-bar entries on oscillator crosses (MACD line/signal, Stochastic K/D, RSI midline). Currently L-type only handles price-vs-price crosses. OC would leverage Hi-Fi infrastructure to recompute oscillators on 1-second bars and detect precise cross moments. Many traders use these as primary entries — currently they can only fire at bar close. Architecturally separate from L (different mechanism) but conceptually a peer execution type.
- **Once OC exists**, the existing oscillator user packs (Stochastic, RSI, MACD-based) will gain a new execution type variant automatically via the registry. No need to refactor the packs themselves — they'll just have one more option in their auto-generated trigger variants.

---

### Milestone 5.6: Time Exit Packs — COMPLETE (2026-04-10)
**Priority:** High — addresses overnight gap risk identified during M5.5 verification
**Status:** Complete. Implemented as a new optimizable variable category (5th pack type), not as exit triggers.

**Why:** Stops and targets fire on price levels. They don't fire on *time*. A position held into market close can be devastated by an overnight gap — the engine correctly produces the gap-down fill, but there's no mechanism to *prevent* the overnight hold in the first place. This is a mechanical risk control, not a signal-driven exit.

**What shipped:**
- **New pack category: Time Exit Packs** — 4th optimizable variable alongside TF Confluence, General, and Risk Management packs
- **4 templates:** End of Day (N min before close), Time of Day (exit at clock time), Max Hold Bars, Session Window (exit outside time window)
- **Engine integration:** Priority 2 in exit chain (after stops, before bail/target/signal/bar_count). Clears pending LC/CC confirmations. Timezone-aware (converts UTC bar timestamps to ET).
- **Full frontend:** 7th column in OptimizableVariables card, Time Exit selector in Strategy Builder, dedicated management page at `/confluence-packs/time-exit` with template→variation nesting, Strategy Detail/Strategies/Mass Builder integration.
- **API:** `GET/PUT /api/packs/time-exit`, `GET /api/packs/time-exit/templates`, `time_exit_pack_id` field on BacktestRequest.
- **DB:** `time_exit_packs` table in Supabase (user_id PK, packs JSONB).
- **Crypto-safe:** `applies_to_24_7` flag per template — EOD/session exits skip for 24/7 sessions, max hold bars still applies.

**Key files:**
- `src/time_exit_packs.py` — data module, TEMPLATES, `check_time_exit()`, persistence
- `src/unified_engine.py` — PSM init, `check_exit` Priority 2, `check_exit_bar_close` Priority 2
- `src/api/routers/packs.py` — time-exit endpoints
- `frontend/src/views/TimeExitPage.tsx` — pack management page
- `frontend/src/views/StrategyBuilderPage.tsx` — OptimizableVariables 7th column + selector

**Tasks (all complete):**
- [x] 5.6a. Backend data module (`time_exit_packs.py`) — dataclass, TEMPLATES, check_time_exit(), persistence, defaults
- [x] 5.6b. DB helpers + API endpoints (`/api/packs/time-exit`)
- [x] 5.6c. Backtest service — pack ID resolution, strategy dict wiring, router helpers
- [x] 5.6d. Engine — PSM init, Priority 2 in both check_exit paths, timezone-aware (UTC→ET conversion)
- [x] 5.6e. Frontend hooks + types (`useTimeExitPacks`, `TimeExitPackDTO`)
- [x] 5.6f. Frontend Strategy Builder, Strategy Detail, Strategies, Mass Builder pages
- [x] 5.6g. Frontend TimeExitPage — V5-style management page with template→variation nesting

**Bug fixed during QA:** EOD exit was comparing UTC hours against ET close time — bars at 15:50 UTC (11:50 AM ET) triggered false exits. Fixed by converting bar_time to `America/New_York` in `_parse_bar_time()`.

**Deferred:**
- Wire Mass Builder time exit checkboxes to actual search execution payload
- Unit test suite (`test_time_exit.py`) for edge cases
- Standardize all confluence pack management pages to consistent V5 nesting design

---

### Milestone 6: Strategy Builder & Strategy Detail Polish — COMPLETE (2026-04-10)
**Priority:** High — strategies are the core product
**Status:** Complete. All core polish items verified and shipped.

**What shipped:**
- **6a: Chart display** — boolean/string column filtering and candle color overrides verified working in `buildStrategyChartPanes.ts`
- **6b-6c: Strategy Builder flow** — all 5 pack types (TF Confluence, General, Risk Management, Time Exit, User Packs) work end-to-end
- **6d: Strategy Detail** — wired btStart/btEnd, extended KPIs from `useStrategyKPIs`, time exit display (orange badge), exit reason colors for time exit reasons, Performance vs Plan placeholder when < 3 forward trades
- **6e-6f: Forward testing / alerts** — backend verified working; frontend displays basic KPIs. Full alert QA deferred to M8.5 (requires live data).
- **Strategy save format fix** — stored_trades now uses raw API trades (ISO timestamps, snake_case), raw kpis, and equity_curve_data. Pack IDs resolved to inline configs at save time.
- **Time Exit analysis tab** — 7th tab in analyzer card, compares all enabled time exit packs + "No Time Exit" baseline
- **Timeframe selector** — driven by user settings (Timeframes pack `primaryEnabled`), supports sub-minute bars
- **Confluence heatmap PB shift** — PB conditions show previous bar's state (what the engine checked), CB shows current bar's state. Labels include [PB]/[CB] fidelity suffix.
- **Heatmap fidelity documentation** — CB backtest↔live divergence risk documented in `project_hifi_confluence_conundrum.md`

**Deferred items:**
- Daily ROI calculation — needs portfolio-level risk context (not a strategy-level metric)
- Chart scroll/zoom bounds — cosmetic, can wait
- Live chart on Strategy Detail — deferred to M8.5
- Hi-Fi confluence / 4-color heatmap — future phase, documented
- Equity curve fidelity badge (Standard vs Hi-Fi label)
- Confluence pack management page standardization (V5 nesting design across all pack pages)

---

### Milestone 7: Mass Strategy Builder — ✅ COMPLETE 2026-04-12

**Priority:** High — required for scale testing
**Effort:** 1-2 weeks (actual)

Wire the Mass Strategy Builder to work with the unified engine and user packs. This is the tool that enables testing hundreds of strategy variations in one batch.

**Tasks:**
- [x] 6a. Verify Mass Builder works with unified engine (not legacy `generate_trades()`)
- [x] 6b. Wire user pack triggers into Mass Builder trigger selection
- [x] 6c. Verify Mass Builder results are consistent with individual Strategy Builder backtests
- [ ] 6d. Wire Hi-Fi mode into Mass Builder — **deferred to M8.6 (Hi-Fi Confluence milestone)**
- [x] 6e. Bulk save: save top N strategies from Mass Builder results
- [x] 6f. Progress tracking: show real-time progress during mass backtest runs
- [ ] 6g. Performance optimization: parallelize mass backtests — **deferred to M9 (tasks 9l-9r)**
- [x] 6h. Wire TF Confluence and General Confluence selections to search scope (allowed_labels filter)
- [x] 6i. Per-state TF Confluence UI (individual state checkboxes grouped by Bull/Bear/Neutral)
- [x] 6j. Preview metric split: Trigger Backtests + Confluence Backtests, with calibrated time estimate
- [x] 6k. Save-by-reference (fixed wrong-strategy-saved after sort change)
- [x] 6l. Mass Results page: preserve row-level status, wire Edit/Delete/Cancel buttons
- [x] 6m. Mass Builder `?edit={id}` handler — loads saved config + results, auto-suffixes name on re-run
- [x] 6n. Live progress on Mass Results page (per-row via useMassProgress)
- [x] 6o. Two-tier progress bar — phase pill (Loading/Preparing/Backtesting/Searching confluences) + sub-second bar/combo counter
- [x] 6p. Cancel/Delete while running: signal cancellation through to worker thread; propagate `_CancelledError` past all catch-alls

**Exit Criteria met:** User can run 100+ strategy variations through Mass Builder, results match individual backtests, top strategies can be saved in bulk (correctly, regardless of sort order), TF/General confluence selections constrain the search space, cancel/delete cleanly stops in-flight work.

**Calibrated constants (from two real runs):**
- Trigger BT: ~7043ms + 343µs × bars
- Confluence BT: ~0.9ms per combo
- Overhead: ~9s per (ticker, TF) group + 397µs × total bars loaded

**Design doc for reference:** `docs/Mass_Builder_Progress_Design.md`

---

### Milestone 8: Portfolios Polish
**Priority:** High — portfolios combine strategies into tradeable units
**Effort:** 1-2 weeks
**Status:** ✅ **Portfolio Detail page fully wired 2026-04-21** (Phases A-D of `distributed-squishing-backus` plan). All live-monitoring surfaces render real data. Remaining work (Anomaly Detection tab, New Portfolio creation-page polish, dollar-weighted Performance vs Plan upgrade) is explicitly deferred per Kevin's direction — tracked in this roadmap as items 9w (dollar-weighted PvP) and below.

**Shipped 2026-04-21:**
- Live Dashboard: KPI row (alert-sourced), Open Positions (Notional + Risk split), Buying Power Tracker (current balance from Account ledger, 24h intraday chart via new `BuyingPowerTimeline.tsx`), Performance vs Plan chart (R-multiple port reusing `PerformanceVsPlan.tsx` with visibility-filtered strategies), Trade History (P Qty / E Qty columns with Phantom/Matched/Live/Backtest status badges, pagination).
- Webhook Delivery History: wired to `/api/webhooks/delivery-log` filtered to portfolio's strategies.
- "Update All Data" bulk refresh: wired on My Portfolios list (dedups strategy IDs across portfolios, per-strategy failure surfacing). Relabeled matching button on My Strategies for consistency.
- Shared `visibility` state hoisted to parent so Strategies tab toggles affect Live Dashboard.
- Sparkline empty-state clarity ("No trades yet — run Update All Data..." instead of flat zero line).
- Balance calc consistency: Current Balance on both Live Dashboard and Account tab uses `starting_balance + deposits − withdrawals` (user-maintained working capital).
- Recurring fix: NaN/Inf in JSON-serialized responses — sanitized list/preview endpoints (commit 501f28b for `/api/strategies`, 70fdec3 for `/api/portfolios/{id}/trades`). Add to standing review checklist for any endpoint returning computed KPIs.

**Operational fixes shipped alongside M8 (same day):**
- Supabase auth resilience: 5-min JWT validation cache + graceful fallback when Supabase 5xx/times out (commits c289cf0, 8019793). Prevents single-worker lockup during Supabase outages.
- API startup no longer blocks on Mass Builder orphan cleanup (commit 82e5816) — moved to background thread.

**What's wired (uncommitted on `dev` as of 2026-04-13):**

Backend (`src/api/routers/portfolios.py` +480, `src/portfolios.py` +18, `src/alerts.py` +182):
- `GET /api/portfolios?preview=true` — enriches each portfolio with kpis, downsampled equity curve preview (~30 points), fwd_kpis, alert_kpis, requirement set pass ratios
- `POST /{id}/worst-case` — worst day, streaks, rolling DD, breaches
- `POST /{id}/capital-utilization` — peak capital deployed timeline, insufficient-capital events (backtest or alerts source)
- `GET /{id}/open-positions` — alert-derived open positions
- `POST /preview` — real-time KPI + equity curve for unsaved portfolio payloads (Portfolio Builder)
- `POST /recommendations` — scores candidate strategies by timeframe diversity, exec type diversity, PF, win rate
- `/compute` extended with `equity_curve_with_strategies`, `benchmark`, `drawdown` include options; returns `starting_balance`
- `_sanitize_for_json()` — fixes numpy/pandas NaN/inf serialization crashes
- `trades_per_day` KPI added; requirement evaluation returns `type`/`threshold`/`actual_raw`
- Webhook Groups scaffolded in `alerts.py` — 11 event types (entry/exit/cancel market+limit, compliance_breach), CRUD functions in place, `webhook_groups_router` registered in `main.py` **but router endpoints not implemented yet**

Frontend:
- **PortfolioDetailPage** (~1183 LOC changed) — Performance tab fully real-data: combined equity curve + per-strategy overlays (legend capped 8), Drawdown Analysis with DD limit reference line, Daily P&L Distribution histogram, Capital Utilization bar chart w/ balance threshold, Daily P&L vs Limits (color-coded), Worst Case cards, Monte Carlo tab with shuffle mode + n_simulations controls. Account tab: balance metrics, balance history chart, deposit/withdraw forms wired to ledger mutations, ledger table with delete. Prop Firm Check tab renders daily loss & pause rules with progress bars.
- **PortfolioNewPage** — synthetic preview replaced with `usePortfolioPreview()`; recommendations engine wired; filters out already-selected
- **PortfoliosPage** — `usePortfolios({ preview: true })` drives real `MiniEquityCurve` sparklines + real fwd/alert KPIs; status heuristic from req-set pass ratio or PF
- **usePortfolios hooks** — added `usePortfolioWorstCase`, `usePortfolioCapitalUtilization`, `usePortfolioOpenPositions`, `usePortfolioRequirementsCheck`, `usePortfolioPreview`, `usePortfolioRecommendations`
- **Sidebar** — added `/alerts/webhook-groups` nav entry (page not yet built)

**Task status:**
- [x] 7a. Portfolio KPI aggregation with user pack strategies — verified via `calculate_portfolio_kpis`
- [x] 7b. Portfolio equity curve correctly combines strategies — `equity_curve_with_strategies` wired + rendered
- [ ] 7c. Monte Carlo with user pack strategies — endpoint + UI exist, **not yet tested with user-pack strategies**
- [ ] 7d. Buying power & compliance rules — capital utilization endpoint complete, UI mostly wired, needs full verification
- [ ] 7e. Anomaly detection — **untouched**
- [ ] 7f. Live dashboard real data — LiveDashboardTab still has hardcoded sample data
- [~] 7g. Portfolio bug fixes — JSON sanitization fix done; ongoing as issues surface
- [x] 7h. Portfolio creation from strategy selection — PortfolioNewPage fully wired w/ preview + recommendations

**Cutoff point:** PortfolioDetailPage Webhooks tab has comment *"Delivery History (placeholder — requires live alert data, deferred to M8.5)"*. Template-filtered `/delivery-log?template_id=*` query param exists in backend but UI is not wired to it. Webhook Groups router endpoints are the other half-finished thread.

**Today's additional work (2026-04-14 — committed in `00f2450`):**
- Webhook Groups CRUD router implemented + management pages at `/alerts/webhook-groups`
- Test delivery now renders `{{placeholders}}` via `render_payload()` before send
- `send_webhook()` no longer retries on HTTP 4xx (fixes duplicate payload dispatches)
- Invalid JSON templates return a clear error instead of wrapping as `{content: ...}`
- `webhook_group_id` persisted on portfolios (added to `_PORTFOLIO_NON_DB_FIELDS`)
- Shared-URL-mode-aware status dots; configured-count consistency across 3 surfaces
- Webhook test data JSON (missing-comma typos fixed)
- Kevin verified all 11 event types dispatch correctly in test mode

**Open threads — revised priority (2026-04-14):**

*Next up:*
1. **PortfolioNewPage end-to-end verification with user-pack strategies** — Kevin hasn't yet created a portfolio using new-system strategies. Verify the full Create → Preview → Recommendations → Save → Appears-in-List → Opens-Detail flow with at least one user-pack strategy before moving on.
2. **M8.5 (Live Data & Real-Time Charts)** — start next after #1. Live data will make M8 verification dramatically easier.

*Deferred until after M8.5:*
- 7c Monte Carlo verification with user-pack strategies
- 7d Buying power / compliance-rule verification walk-through
- 7e Anomaly detection wiring (untouched)
- 7f LiveDashboardTab real data (depends on M8.5 infrastructure anyway)
- 7g General portfolio bug sweep
- Webhook Delivery History UI on PortfolioDetailPage (depends on live alert data)
- "Design Ref" tab cleanup on PortfolioDetailPage

*Deferred beyond milestones (Kevin's call):*
- Compliance-breach multi-symbol close flow — decide between (A) compliance_breach event loops through open positions per symbol, or (B) use existing exit_*_market/limit events for actual close orders and keep compliance_breach as notification-only. Recommendation: B.

**Tasks:**
- [ ] 7a. Verify portfolio KPI aggregation works with user pack strategies
- [ ] 7b. Verify portfolio equity curve correctly combines multiple strategies
- [ ] 7c. Verify Monte Carlo simulation works with user pack strategies
- [ ] 7d. Verify buying power tracking and compliance rules
- [ ] 7e. Verify anomaly detection works correctly
- [ ] 7f. Portfolio live dashboard: wire real data (currently placeholder in some areas)
- [ ] 7g. Fix any portfolio-level bugs surfaced during testing
- [ ] 7h. Portfolio creation from strategy selection (select strategies → create portfolio)

**Exit Criteria:** User can build portfolios from strategies (including user pack strategies), view aggregated performance, run Monte Carlo simulations, and track portfolio health — all working correctly.

---

### Milestone 8.5: Live Data & Real-Time Charts
**Priority:** High — required for alert verification and live trading confidence
**Status:** Phases B AND B+ **shipped 2026-04-14**. Sub-minute live data working. Architecture proven end-to-end. Plan: `~/.claude/plans/reflective-riding-sifakis.md`. Backup branch: `dev-backup-pre-m8.5`.

**Phase B+ shipped (commits d623db7, af75b28, feaef8f, 0352eb7, df345ef):**
- `BarBuilder.accept_second_bar()` aggregates per-second bars into N-sec primary bars with bar-for-bar parity to `data_loader.resample_to_timeframe()` — locked by `test_ralph_subminute_parity.py` (6 cases)
- Sub-minute strategies (10Sec, etc.) now receive live bars + alerts for the first time
- Per-second forming-bar broadcasts every ~250ms for all primary TFs (1Min+ uses partial for chart visual; AM remains canonical for indicators/triggers)
- `A.{ticker}` Polygon channel subscribed for all stocks (was gated behind never-populated `_intrabar_triggers`); subscription went 13 → 26 channels
- `_intrabar_triggers` populated from resolved entry/exit triggers — restores L-type live trigger detection (silently broken before)
- Monitor pipeline factored into shared helper `_run_monitor_pipeline_for_completed_bar` so AM + sub-minute paths produce identical post-bar handling
- `chartPanes` memoized in StrategyDetailPage (also fixes a CLAUDE.md IIFE rule violation)
- `SyncedChartPane` refactored to persistent-instance pattern: `paneStructureKey` JSON hash distinguishes structure changes from data changes; data updates use `series.setData()` (preserves visible range); structure rebuilds save/restore visible range. User's pan/zoom now survives indefinitely until they explicitly change pane structure.

**Architecture decision (2026-04-14):** Skipped REST-polling fallback. Single source of truth = Ralph. Forming/completed bars broadcast via Supabase Realtime channel `live_bars:{user_id}`; frontend subscribes via `useLiveBar` hook and updates the chart imperatively via `series.update()` to avoid React re-render cascade at the publish cadence.

**What shipped (commits dfbc7b9, 05de900, 561416e, b7f44e7, 0403d2b, d561226):**
- `src/live_bar_publisher.py` — HTTP-based broadcast publisher with per-(symbol, tf) throttle (250ms forming, immediate completed), try/except isolation, fire-and-forget via `asyncio.create_task`. Verified Supabase returns 202 Accepted on every publish.
- `ralph_engine.py` — `SymbolHub` accepts a publisher; `_publish_completed_bar()` injected after the existing alert dispatch in `on_tick` (~line 941), `flush_stale_bars`, and `on_polygon_bar`. Hot path unchanged for alerts; broadcast is an additive side effect.
- `worker.py` — instantiates publisher from env, threads `user_id` through every monitor.
- `useLiveBar.ts` (frontend) — subscribes to `live_bars:{user_id}` broadcast, filters by (symbol, tfSeconds), returns latest bar + receivedAt timestamp.
- `TradingChart.tsx` + `SyncedChartPane.tsx` — accept optional `formingBar` prop; dedicated `useEffect` calls `candleSeries.update()` imperatively. Zero React re-render cascade.
- `StrategyDetailPage.tsx` Chart & Trades tab — wired with status pill ("● Live" green / "○ Not live" gray, recomputed at 1Hz against a 5×TF window).

**Monitor-everything change (committed alongside):**
- `db.py:get_monitored_strategies_db()` — was filtering to "in a portfolio with active webhook." Now: every strategy with `entry_trigger_confluence_id` is monitored. Webhook routing still respects portfolio + group config at alert time. Aligns with Kevin's stated long-term direction: every saved strategy accumulates forward-test data and broadcasts live bars.

**Tasks:**
- [x] 8.5a. **Live price chart on Strategy Detail** — Ralph-sourced via Realtime broadcast, imperative chart update
- [~] 8.5b. **Alert verification with live data** — pipeline proven (broadcasts firing, pill turning green); full alert-trade verification with accumulated forward-test data is ongoing
- [ ] 8.5c. **Forward testing live QA** — alert/forward parity checks pending data accumulation
- [ ] 8.5d. **Confluence pack pages requiring live data** — deferred
- [ ] 8.5e. **Position Monitor live integration** — same Realtime channel can carry position-state events; layer on next session

**Verified end-to-end (2026-04-14):** Ralph running locally, Railway worker paused via `WORKER_DISABLED=true`. Strategy 47 (NVDA 1Min RTH) shows green "Live" pill on Chart & Trades tab; broadcasts arrive every minute on bar close. Logs confirm `POST /realtime/v1/api/broadcast 202 Accepted` per (symbol, tf) bar.

**Phase B+ in-progress (2026-04-14 — added late in session):**
- ~~Forward-test pipeline gap~~ — **SHIPPED (commit ffc6c1d).** The unified engine's `_signal_exit` now embeds a full `trade_record` in every exit signal. `DBAlertDispatcher._persist_algo_trade()` persists it to `stored_trades` immediately. Frontend `useStrategy`/`useStrategyTrades`/`useStrategyForwardTest` refetch every 60s. Chart & Trades tab shows new algo markers within a minute of an exit firing. Verified live: strategy 71 went from 7,414 → 7,415 trades on first live exit.
- **Indicators/heatmap first-load caching audit — follow-up.** `/api/strategies/{id}/chart-data` endpoint takes ~10s on first load. Deferred to next session.

**Phase B+ close-out (2026-04-14, late evening):**
- Chart lag root-caused to 74,000-node DOM (7,400 trade rows × 10 td cells). Capped Algo History table to last 100 rows with "Show all" toggle → chart interaction responsive (commit c90b465).
- `/api/strategies/{id}/forward-test` endpoint was calling expensive `prepare_forward_test_data()` (fresh Polygon bars + full engine rerun) on every load. Switched to fast in-memory split of cached `stored_trades` — returns in ~2s instead of 30-120s (commit 420a8c8).
- Frontend `useStrategyForwardTest` was gated behind a `fwdRequested` flag that defaulted false, so forward/backtest split never rendered. Removed the gate (commit dc652a5).
- React perf pass: `useChartPrefs` memoized (commit ca297d3), `SyncedChartPane` wrapped in `React.memo` with memoized `formingBar` prop (commit 0f7c5ba), `chartTabData` deps stabilized, markers always re-applied (commits a24a693, e815bfb).
- **Data-loss incident + recovery (commit ab8e954, regression test in 6dd8bcb):** `update_strategy_admin` called `_strategy_to_row()` which emits `config = {}` on partial updates, wiping the JSONB config column. Strategies 70 and 71 had all trigger/stop/target/confluence settings wiped when their first live exits fired. Hotfix: function now splits columns vs. config manually and only includes `config` in the UPDATE payload when config fields are actually being changed. Strategies 70 and 71 manually restored from worker log breadcrumbs (UTBot v2 buy entry, swing stop, bar count 4) — stored_trades history preserved throughout, only JSONB config was affected. 5-case regression test in `test_strategy_partial_update_safety.py` locks the fix. **Lesson:** never use `_strategy_to_row()` for partial updates; it's designed for full-strategy CREATE, not partial UPDATE.

**Deferred follow-ups (out of MVP scope):**
- ~~Ralph live sub-minute bar aggregation~~ — **SHIPPED in Phase B+ (2026-04-14).**
- ~~Forming-bar streaming via `on_second_bar` aggregation~~ — **SHIPPED in Phase B+.**
- Live trade markers on chart (subscribe to alert events via separate broadcast channel)
- Live indicators / oscillator pane updates (publish interpreter state alongside OHLCV)
- Private Realtime channels with RLS (security hardening — current channel-name-by-UUID is acceptable for dev)
- TradingChart.tsx persistent-instance refactor (lower priority; SyncedChartPane is the main surface)
- Theme color hot-update via `applyOptions()` (currently rebuilds chart on theme switch — rare action, acceptable)
- Higher-than-1Min strategies (5Min, 15Min, etc.) — Ralph's `on_polygon_bar` always uses `tf_seconds=60` for AM bars. 5Min strategies get the 60s builder fed but their 300s builder never gets bars in live. Pre-existing limitation; would benefit from a similar `accept_second_bar`-style approach OR aggregating 1-min bars upward.
- `webhook_deliveries` column missing in `alerts` schema cache — pre-existing; alerts fire correctly but webhook delivery logging fails. Out of M8.5 scope.

**Architecture notes:**
- Polygon REST polling (completed bars every N seconds) vs WebSocket (tick-level, 1 connection limit)
- REST polling is simpler and doesn't conflict with Ralph's WS connection
- Consider caching bar data to reduce API calls across multiple strategy detail pages
- Chart should NOT flicker on update (the key reason for migrating from Streamlit)

**Exit Criteria:** Strategy Detail shows a live-updating price chart with trade markers, alerts fire and display in real-time, forward test data accumulates correctly.

---

### Milestone 8.6: Hi-Fi Confluence — Trustworthy CB Fidelity
**Priority:** High — required for backtest↔live parity on Current Bar strategies
**Effort:** 1-2 weeks
**Status:** Not started. Depends on M8.5 (live data needed to verify parity).
**Design documented:** `project_hifi_confluence_conundrum.md`

**The problem:** Current Bar (CB) fidelity confluence creates backtest↔live divergence. In live trading, Ralph evaluates confluence at the trigger moment (mid-bar). In backtests, confluence is only evaluated at bar close. If the state changes mid-bar, the backtest and live engine disagree — creating phantom or missed trades.

Previous Bar (PB) fidelity is safe because the prior bar is fully closed.

**The approach (Kevin's targeted Hi-Fi confluence):**
Rather than recomputing indicators on every 1-second bar, piggyback on the existing Hi-Fi infrastructure:
1. Hi-Fi already loads 1-second bars around each entry to resolve fill prices
2. At the trigger cross moment, also recompute ALL confluence timeframes from 1-second data — not just the primary TF
3. Each TF is resampled from 1-second data up to the trigger timestamp (incomplete forming bars, matching live behavior — no forward-fill bias)
4. If all confluence conditions are met at trigger moment → entry is valid
5. If any condition fails → trade is filtered out

**Why all TFs must be recomputed:**
- Lower TFs (e.g., 1-min confluence on 15-min primary): changes multiple times per primary bar
- Higher TFs (e.g., 1-hour confluence on 1-min primary): has an incomplete forming bar at trigger moment
- Forward-filled states reflect the *previous closed* bar, not the forming bar — wrong for CB fidelity

**Heatmap visualization — "Snap to Entry":**
- Bars WITH entries: color reflects state at trigger moment (what the trader cares about)
- Bars WITHOUT entries: color reflects bar-close state (standard)
- 3-color scheme: Green (met), Red (not met), Yellow (not met at close, BUT met at trigger moment — Hi-Fi confirmed valid entry)

**Tasks:**
- [ ] 8.6a. Extend `_hifi_resolve_trades` to load wider 1-second window for confluence recomputation
- [ ] 8.6b. Resample 1-second data to each confluence TF up to trigger timestamp
- [ ] 8.6c. Run indicator + interpreter pipeline on resampled data at trigger moment
- [ ] 8.6d. Gate entries: filter trades where confluence wasn't met at trigger moment
- [ ] 8.6e. Mark trades with `hifi_confluence_met` field for frontend consumption
- [ ] 8.6f. Frontend heatmap: yellow color for entry bars where bar-close state was red but trigger-moment state was green
- [ ] 8.6g. Verify parity: compare Hi-Fi CB backtest results against live Ralph alert trades

**Exit Criteria:** CB fidelity backtests match live behavior — no phantom or missed trades due to mid-bar confluence changes. Heatmap clearly communicates the distinction via yellow markers.

---

### Milestone 8.7: Live Bar Cache — Unified Data Source for Live + Backtest
**Priority:** Required before live trading at scale
**Effort:** Larger than originally estimated due to scope expansion (data capture infrastructure for engine-truth precision + UI work)
**Status:** **Substantially complete as of 2026-05-02.** Write path + cache backed Lab tab + alert/engine state capture all shipped this weekend. Read path for backtest (M8.7d) is what's left. See `docs/Plan_M8.7_Saturday_2026-05-02.md` for the full milestone-by-milestone record.
**Background:** `docs/Parity_Trust_Roadmap_2026-04-29.md`, `docs/Plan_Live_Bar_Cache_2026-04-30.md`, `docs/Drift_Analysis_2026-04-30.md`

**The problem:** Live worker uses Polygon **WebSocket** aggregated bars; backtest uses Polygon **REST** bars. Drift analysis 2026-04-30 confirmed 79.4% of 96h of entry+C alerts had differing bar close vs REST (median $0.02, p95 $0.09, max $0.835). Polygon's own docs (`massive.com`) explicitly state WS ≠ REST is normal — late-print mechanics differ between feeds. Engine math is correct on both paths, but EMA chains accumulate the small differences, so cumulative state drift causes sensitive triggers to fire differently between live (WS-history) and backtest (REST-history). Net effect: meaningful trade-set divergence between live alerts and a fresh backtest, especially on knife-edge sub-minute strategies.

**The approach:** Have the worker write its WS-aggregated bars to a Supabase `live_bars` table as they're built (`source='ws'`). Backtest can eventually read from this cache, falling back to REST only for bars older than the cache. Live and backtest then operate on identical OHLCV — engine state agrees by construction.

Saturday 2026-05-02 expanded scope to also discover + fix a duplicate-bar bug in BarBuilder (Polygon WS rebroadcasts within 15-min FINRA window were being treated as new bars instead of corrections), plus build the Lab tab visualization, models placeholder, alert snapshots, and engine-state capture infrastructure.

**Schema (shipped):**
```sql
-- live_bars: per-bar OHLCV with first-write + latest-write columns
CREATE TABLE live_bars (
    symbol TEXT, timeframe_seconds INT, bar_start TIMESTAMPTZ,
    open, high, low, close, volume DOUBLE PRECISION,
    first_open, first_high, first_low, first_close, first_volume DOUBLE PRECISION,
    written_at TIMESTAMPTZ, last_updated_at TIMESTAMPTZ,
    source TEXT,  -- 'ws' or 'rest_backfill'
    PRIMARY KEY (symbol, timeframe_seconds, bar_start)
);

-- bar_engine_states: per-bar full indicator state per strategy (engine-truth precision)
CREATE TABLE bar_engine_states (
    strategy_id INT, bar_start TIMESTAMPTZ,
    indicator_state JSONB,
    source TEXT, written_at TIMESTAMPTZ,
    PRIMARY KEY (strategy_id, bar_start)
);
```

**Tasks (Required, in order):**
- [x] 8.7a. [Required] Schema + migration for `live_bars` table — ✅ shipped 2026-04-30
- [x] 8.7b. [Required] Worker write path (fire-and-forget) — ✅ shipped 2026-04-30
- [x] 8.7c. [Required] Validation script (`_validate_live_bars_cache.py`) — ✅ shipped 2026-04-30
- [x] 8.7-first. [Required] Add `first_*` columns + DB triggers preserving first-write values across upserts — ✅ shipped 2026-04-30
- [x] 8.7-rebroadcast-fix. [Required] **Duplicate-bar bug fix** (M1 of Saturday's plan) — `accept_bar` + `accept_second_bar` detect Polygon WS rebroadcasts and replace history rows; new `IncrementalIndicatorEngine.recompute_from_history()` resets state then replays. Replay test (`test_rebroadcast_recompute.py`) validates Option B is mathematically equivalent. ✅ shipped 2026-05-02 (commit `a15461f`).
- [x] 8.7-lab-phase2. [Required] **Lab tab Phase 2** (M2) — `prepare_data_with_indicators` accepts pre-loaded DataFrames; new `/chart-data-cache` endpoint computes indicators+heatmap from cache; Alert Lens right side shows what live engine actually sees. ✅ shipped 2026-05-02 (commit `7bd71e7`).
- [x] 8.7-models-placeholder. [Required] **Models-as-strategy-variable** (M3) — `backtest_model` + `live_model` declared per strategy. Schema + UI shipped; engine dispatch on selection comes with M8.7d. ✅ shipped 2026-05-02 (commit `8b40eff`). Documented in `docs/Strategy_Models_2026-05-02.md`.
- [x] 8.7-alert-snapshots. [Required] **Alert indicator snapshots** (M4) — Worker captures `monitor.indicators.state.current` at fire moment. Stored in `alerts.data.indicator_snapshot` JSONB. Frontend renders 📊 tooltip. ✅ shipped 2026-05-02 (commit `064f9a3`).
- [x] 8.7-replay-mode. [Required] **Lab tab bar-by-bar replay** (M5) — New `ChartReplayCard` reusing existing ReplayableChart + ReplayControls. ▶ Replay toggle on Alert Lens. ✅ shipped 2026-05-02 (commit `064f9a3`).
- [x] 8.7-engine-state-capture. [Required] **Engine-truth state capture foundation** (M6) — `bar_engine_states` table + `bar_engine_state_writer` (env-flag gated, fire-and-forget). Pure data-capture today; no read path yet. ✅ shipped 2026-05-02 (commit `064f9a3`).
- [ ] 8.7d. [Required] **Backtest read path** — extend `load_market_data` (or service variant) to check `live_bars` first, REST fallback for gaps. Opt-in initially via flag. **NEXT SESSION** (Sunday or next week).
- [ ] 8.7e. [Required] Gap detection + REST backfill — cache becomes self-healing
- [ ] 8.7f. [Required] Cutover — flip `load_market_data` default to cache-first
- [ ] 8.7g. [Required] Re-run parity sweep + Q3 fidelity drill — confirm fidelity ~100% on cached-period strategies
- [x] 8.7-engine-dispatch. [Required] **Phase C — Live engine dispatch** ✅ shipped 2026-05-05 (commit `74ff56c` + `f971eeb` + `b1ad5c0`). `StrategyMonitor.live_model` declared from config; symmetric source-label gate in `_run_monitor_pipeline_for_completed_bar`; `WsAggMinuteBuilder` dispatches `ws_agg`-source 1Min bars; A.* subscription gate expanded with `has_ws_agg`. Default flipped to `ws_agg_locked`; all 39 strategies bulk-migrated. Live-verified 2026-05-05 (strategy 166 round-trip + 8/8 bit-identical closes vs Polygon REST) and 2026-05-06 RTH open (META/TSLL/SPY/TSLA: 100% ws_agg coverage; AAPL/AMD AM-only ~30% as predicted). See `docs/Live_Model_Decision.md`.
- [x] 8.7-tv-validation-2026-05-06. [Validation] **TradingView vs ws_agg comparison** ✅ 2026-05-06. SPY 1Min: 300 paired bars, close 99.7% bit-identical (max diff $0.03), open 94.7% bit-identical. SPY 10Sec (`ws` source — same A.*-aggregation as `ws_agg` but at sub-minute): close median diff $0.0299 (different aggregator: TV=CBOE/TRF, us=Polygon SIP); structural, not a bug.
- [x] 8.7-algo-history-incremental. [Required] **Algo-history incremental writer** (5-min cron, 15-min lag) ✅ shipped 2026-05-06 (commits `b0b7fe96` + dual-write removal). Append-only via set-diff on `(entry_fill_ts, exit_fill_ts)`. Cron daemon thread in WorkerManager.
- [x] 8.7-windowed-engine. [Required] **Windowed engine helper** (Phase 1) ✅ shipped 2026-05-06 (commit `0192173`). Ports Streamlit's `_generate_incremental_trades` to FastAPI; cron now uses windowed path on stamped strategies — drops engine cost from ~30-40s to ~3-5s per strategy.
- [x] 8.7-background-jobs. [Required] **Background job pattern + Jobs page** ✅ shipped 2026-05-06 (commits `5f21e16` + `314b176`). Mirrors Mass Builder pattern; in-memory state. `/jobs` page lists progress with per-strategy detail, cancel, skip-reason, engine-scope.
- [x] 8.7-cache-locked. [Required] **Phase E preview — cache_locked dispatch** ✅ shipped 2026-05-06 (commit `d06f992` + Hi-Fi expansion `54da0f5`). `services._resolve_primary_df_for_backtest_model` routes cache-backed models. `cache_locked.available=True`. Lab tab filters `sources=['ws','ws_agg']`. Pre-staged sids 152 + 88 as canaries.
- [x] 8.7-builtin-template-cleanup. [Required] **Built-in template cleanup** ✅ 2026-05-07. Deleted 11 strategies (sids 6,8-17) on legacy built-in templates (`ema_price_position_v2`, `utbot_v2`, `macd_line`, `ema_stack`, `macd_histogram`, `rvol`). All remaining 20 strategies use user-pack triggers exclusively. Built-in templates remain in `confluence_groups.TEMPLATES` for legacy reference but no active strategy references them.
- [x] 8.7-reset-alerts-bulk. [Polish] **Bulk Reset Alerts** ✅ shipped 2026-05-06 (commit `fd49cea`). Replaced dead global button with per-selection action under bulk toolbar.
- [x] 8.7-tv-validation. [Validation] **2026-05-07 RTH alignment validation** ✅ post-Phase-E preview: cache_locked closed trade-COUNT divergence (sid 88: 22 algo vs 21 alert pairs ~1:1, was 80 vs 39 = 2:1 on REST). Entry drift bit-perfect (median 0.0s, max 0.0s). C-type exits bit-perfect. L-type stop_loss/target exits Hi-Fi-refined to sub-second. **Open: L-type signal exits (`*_ib` cross triggers) still bar-aligned — Hi-Fi extension queued (see 8.7-hifi-signal-exits).**
- [x] 8.7-hifi-signal-exits. [Required] **Hi-Fi signal-exit refinement (Phase 1 — static-level crosses)** ✅ shipped 2026-05-07 (commit `169df09`). Extends Hi-Fi Pass 2 to refine L-type SIGNAL exits (e.g. `eppv4_cross_mid_down_ib`) using user-pack `trigger_levels` manifest declarations. New `_walk_1s_for_level_cross` walker in `backtest_service.py`; `pack_registry.get_trigger_level_spec()` resolver; `bar_df` plumbed into `_hifi_resolve_trades`. Covers `eppv3`, `eppv4`, `utv4` (4 packs declare static-level cross semantics). Phase 2 (indicator-vs-indicator, value-vs-threshold, dynamic-level) deferred — see new milestone below.
- [x] 8.7-ai-pack-guardrail. [Required] **AI pack-creation guardrail (`trigger_levels` enforcement)** ✅ shipped 2026-05-07 (commits `508cb89` + pack_builder strengthening). Three-layer guard: (1) `pack_spec.audit_trigger_levels()` regex-detects cross-style trigger names and warns when both `trigger_levels` and `trigger_levels_phase2` are absent; (2) install-time warnings printed during `pack_registry.scan_and_load_all`; (3) AI builder treats audit warnings as errors in `validate_parsed_response`. Manifest schema gains `trigger_levels_phase2` placeholder for intentional non-static markers — silences audit while documenting intent. Cleanup pass added markers to 7 existing packs (ema_stack_v2, macd_histogram_v2, macd_line_v2, rsi_zones, rsi_zones_2, rsi_zones_3, stochastic_oscillator); audit count went from 30+ warnings to 0.
- [ ] 8.7-hifi-signal-exits-phase2. [Future] **Hi-Fi Phase 2 — dynamic + indicator-vs-indicator crosses.** Three evaluator shapes needed: `indicator_vs_indicator` (MACD line vs signal, EMA-vs-EMA, K-vs-D), `value_vs_threshold` (RSI vs 70/30/50, MACD-line vs zero), `price_vs_dynamic` (VWAP intra-bar). Plan-doc note: Kevin proposed introducing a new exec_type **X** (or **D**) for these cases instead of overloading **L** — surfaces the semantic difference at the trade-record level and lets future packs declare exec_type='X' for the engine to route to the correct Hi-Fi handler. Decision deferred until Phase 1 has soaked. See plan-doc and `feedback_exec_type_x_for_indicator_cross.md`.
- [ ] 8.7-cron-throughput-starvation. [Investigate] **Cron starvation on heaviest strategies.** Observed 2026-05-07: oldest-stamped strategies (sid 152/153/154) sat unprocessed by cron for 2-12 hours despite oldest-first sort. Manual run on each succeeded in 28-87s. Hypothesis: heavy strategies (153=87s, 154=82s) consume per-cycle 240s budget before cron loops to remaining queue, AND something about the cycle frequency vs. budget keeps them perpetually at top. Net 267 trades recovered manually. Action: surface budget exhaustion + per-strategy elapsed in cron logs; consider per-strategy round-robin cap OR raising budget. Non-blocking once "Update New Data" page exists as user-driven recovery.
- [ ] 8.7-silent-strategies. [Investigate later] Mass-builder mirrors firing 0 alerts/7d. Likely overly-restrictive confluence; non-blocking.
- [ ] 8.7-strategy-144-card-bug. [Polish] Card alerts count diverges from detail page count. Cosmetic.
- [x] 8.7-phase41-algo-lane-fix. [Required] **Algo lane was wiping backtest data + writing NULL data_source** ✅ shipped 2026-05-11 PM (commit `6b4fc53`). Two writers had the bug:
  - `recompute_and_persist_algo_trades`: DELETE was unfiltered (eq strategy_id only) — wiped `backtest_<model>` rows the backtest lane wrote moments earlier in the same bulk job. INSERT via `trades_store.insert_trade` did not tag `data_source`. Now uses `replace_trades_admin` with `data_source_filter='cache_%'` and tags each trade with `cache_<algo_model>`.
  - `append_new_trades_for_strategy` (cron + forward-test mode): `load_trades_for_strategy` had no data_source filter so set-diff compared new algo trades against backtest rows too (could falsely dedup). `insert_trade` did not tag data_source. Now loads with `cache_%` filter and tags inserts with `cache_<algo_model>`.
  - **Symptom:** sid 152 + sid 154 (the two strategies Update All Data ran on earlier today) ended up with ALL-NULL data_source — breaking the Divergence tab's `backtest_%` / `cache_%` filter joins.
  - **Verified working:** evening batch on remaining strategies produced properly-tagged rows. Final distribution: 34,849 `backtest_rest_hifi`, 11,286 `cache_cache_locked`, 478 legacy `cache_locked`, 1,695 NULL orphans.
- [x] 8.7-supabase-reliability-fixes. [Required] **Writer + worker hardening for Supabase intermittent 522s** ✅ shipped 2026-05-11 PM (commits `d058653` + `78d5597`).
  - `db.replace_trades_admin` + `db.insert_trade_admin` wrap in `_execute_with_retry` (transient errors: Cloudflare 5xx, "JSON could not be generated", connection/timeout; permanent errors: unique/FK/syntax skip retry). 4 attempts, 2s/4s/8s backoff.
  - Bulk INSERT now chunked at 250 rows/call. Smaller payload + per-chunk retry contains failure blast radius.
  - DELETE+chunked-INSERT pattern is idempotent on full retry — second DELETE matches nothing, INSERT runs on second try. Survives Supabase blips that previously corrupted sid 154 mid-write.
  - Worker `db_write_status` throttle: was firing every PICKLE_WRITE_INTERVAL (~2s) ignoring the existing `HEARTBEAT_INTERVAL=30s` constant. Now writes only on state-flip (running/connected) or every 30s. ~15x reduction in monitor_status writes (~1800/hr → ~120/hr per active user).
- [x] 8.7-hifi-data-wipe-fix. [Required] **Hi-Fi UPDATE was destroying trade `data` JSONB on every refresh** ✅ shipped 2026-05-12 (commit `976a041`). `run_hifi_pass2` persist loop called `orig.get('data') or {}` where `orig` came from `load_trades_admin` → `_row_to_trade` (which SPREADS data JSONB into top-level keys). With no `data` key on the flat dict, fallback `{}` was used, merged dict only contained the new `hifi_resolved: true`, and UPDATE wrote that single field — wiping `bars_held`, `hold_time_seconds`, `pnl`, `win`, `behavior`, `confluence_records`, `entry_trigger`, `exit_trigger`, `stop_price`, `target_price` on every row Hi-Fi touched. Fix: reconstruct `orig_data` from non-column fields on the flat dict (those ARE the fields that were originally in the JSONB before unpacking). Existing rows that already lost data need a fresh full recompute to repopulate; going forward, Hi-Fi preserves all JSONB content.
- [x] 8.7-chart-and-trades-wiring. [Required] **Chart & Trades modules now read real algo-lane data** ✅ shipped 2026-05-12 (commit `976a041`). After Phase 41 cutover, the "Algo History" module and "Price Divergence (Algo vs Alert)" module were silently pulling `[...fwdTrades, ...btTrades]` (backtest data) since `stored_trades` hydration filters to `backtest_%`. Now:
  - New endpoint `GET /api/strategies/{id}/algo-trades` returns `cache_%` rows from trades table
  - New hook `useStrategyAlgoTrades` (60s poll)
  - Algo History module reads `algoTrades` instead of merged BT+FWD
  - Price Divergence matching now compares alerts against algo lane (real algo↔alert pairs, not phantom backtest↔alert)
  - `TradeDTO` extended to include hold_time_seconds, pnl, data_source, entry_fill_ts, exit_fill_ts, hifi_resolved, behavior, exit_trigger
- [x] 8.7-admin-divergence-summary. [Required] **Admin divergence summary page** ✅ shipped 2026-05-12 (commit `976a041`). New page at `/admin/divergence` with date range pickers (today/yesterday/7d quick buttons + manual datetime). Sortable table of all strategies showing backtest/algo/live counts, 3-way matches, entry RC drift med/max, exit CL drift med/max. Color cells green ≤2s / yellow ≤30s / red >30s matching the per-strategy Divergence tab. Click sid → /strategies/{id}?tab=divergence. New endpoint `GET /api/strategies/admin/divergence-summary?start=X&end=Y` loops user strategies and runs `compute_three_way_divergence` per strategy in the window.
- [x] 8.7-divergence-tab-polish. [Required] **Divergence tab: memory bounds + date-window filter + sort desc + lane-status clarity + algoMatches stale-closure fix** ✅ shipped 2026-05-12 (commits `4b57710` + `66e1cf8` + `27802ba` + `c2f7713`). API OOM mitigation via `max_per_lane` caps on `/divergence-data` + `/admin/divergence-summary` (with `gc.collect()` between strategies in admin loop). Date window default 24h with quick buttons. Lane-update messages clarified. Algo History Δ + Lab Price Divergence pairs restored via one-line deps fix on `algoMatches` useMemo (line 1927) — Phase A had repointed the body to `algoTrades` but missed deps array.
- [ ] 8.7-secondary-tf-cache-write. [Bug — open as of 2026-05-12 EOD] **Worker secondary-TF cache writes are mostly broken.** SPY 15Min cache today: 1 bar (only 18:30:00). 5Min and 1Hr: 0 bars. Yesterday's 15Min: 5 bars (expected ~26 for a market day). Symptom: algo lane (cache_locked) stops emitting trades when a multi-TF confluence requires a secondary TF — gate freezes on stale state. Sid 154 algo lane stuck at 18:44 UTC today because its `15m-UT_BOT_V4-BULL_TREND` confluence couldn't see post-18:30 bars. NOT caused by today's commits — none touched worker code. Code: `ralph_engine.py` secondary-TF write path (around line 1931-1932 per earlier exploration). Diagnostic: `project_secondary_tf_cache_gap_2026-05-12.md`. Tomorrow's investigation: trace why the secondary-TF builder fires only a handful of times per day instead of every period boundary.
- [x] 8.7-bt-append-true-append. [Required] **BT-APPEND true-append optimization** ✅ shipped 2026-05-12 (commit `67675fb`). `append_new_backtest_trades_for_strategy` was doing a full DELETE+INSERT-all-merged on every refresh, taking 60s+ per strategy with 6,000 backtest rows. Now: (1) single MAX(entry_fill_ts) query for the append anchor, (2) engine windowed compute on `[max_ts, now-lag]`, (3) filter engine output to entries strictly newer than anchor, (4) per-row `insert_trade_admin` (handles unique violations + retries transient errors), (5) re-read all backtest_% rows once for fresh KPIs over the full set, (6) UPDATE strategy row with KPIs + equity_curve_data only. Expected: 6-10× speedup, ~40min/batch → ~5min. New `db.get_max_entry_ts_admin` helper. Algo-lane append (`append_new_trades_for_strategy`) deferred to W2C — already uses per-row inserts so it's structurally fast on the cron-cadence path.
- [ ] 8.7-post-migration-tasks. [Open as of 2026-05-11 EOD] Remaining follow-ups now that Phase 41 + algo-lane fix are live:
  - [ ] **Cosmetic:** retag `cache_cache_locked` (11,286 rows) → `cache_locked` for consistency with legacy convention. Double-prefix is from `f'cache_{algo_model}'` where `algo_model='cache_locked'`. Currently functional via `cache_%` LIKE filter. Either fix the writer to strip the `cache_` prefix OR retag in DB. Non-blocking — Divergence tab works.
  - [ ] **Cleanup:** delete or retag 1,695 orphan NULL `data_source` rows (1,207 on sid 154, 320 on sid 152, plus stragglers). These survived because the fixed algo-lane DELETE only matches `cache_%`. Easiest path is re-running Update All Data on sid 152 + 154 with a one-time DELETE-NULL pre-pass.
  - [ ] Decide whether to re-enable cron (`ALGO_HISTORY_CRON_ENABLED=true` on Railway worker) — fixed writer paths should make it safe now.
  - [ ] Track per-strategy REST↔CACHE entry drift across all 19 strategies; flag any with drift > 10s on 1Min as fidelity issues.
  - [ ] Eventually deprecate `strategies.stored_trades` JSONB column (writes are skipped; column is vestigial). Defer until 1+ week of stable operation.
  - [ ] Re-investigate Hi-Fi Phase 2 (X exec_type for indicator-vs-indicator crosses) — exit drift (~7min median on sid 152) is the remaining fidelity gap. With backtest lane now reliable, Phase 2 has a clearer baseline.
- [x] 8.7-divergence-tab-v1. [Required] **Divergence tab v1 — 3-lane comparison** ✅ shipped 2026-05-07 (commit `cd58026`). New "Divergence" tab on Strategy Detail compares `stored_trades` (Backtest), trades-table (Algo), alerts table (Live). Greedy 3-way joiner with 5min tolerance gate, drift KPIs (median/p95/max), color-coded ≤2s green / ≤30s yellow / >30s red. Config-drift warnings via lane status badges showing model per lane.
- [x] 8.7-algo-model-split. [Required] **algo_model field decouples cron from backtest** ✅ shipped 2026-05-07 (commits `3bf6911` + `a6d0bb8` + `f5d1f69`). New `algo_model` config field: cron uses it instead of `backtest_model`. Bulk migration set all 20 strategies to `backtest_model=rest_hifi`, `algo_model=cache_locked`, `live_model=ws_agg_locked`. Defaults flipped: backtest `rest_only` → `rest_hifi`. Both alert dispatchers (worker + ralph_engine) stamp `live_model` on alert payload at fire time; legacy unstamped alerts render "unknown" in orange. ModelsCard adds 3rd dropdown.
- [x] 8.7-lane-mode-matrix. [Required] **All 4 lane×mode update combinations + JWT-safe endpoint** ✅ shipped 2026-05-08 (commits `d867e63` + `5fddf86`). Two new helpers: `append_new_backtest_trades_for_strategy` (forward → stored_trades JSONB under backtest_model) + `recompute_and_persist_algo_trades` (full → trades table under algo_model). `/update?mode=all|new` wraps lane calls in `set_admin_user_context` to immunize against user JWT expiration during multi-min runs. Strategy Detail header replaces single Refresh with Update New + Update All. Bulk job worker fans both lanes per Option A.
- [x] 8.7-builder-uis. [Required] **Strategy + Mass Builder 3-dropdown picker** ✅ shipped 2026-05-08 (commits `e7bf21a` + `0073dbb`). Strategy Builder gains Models card with 3 dropdowns (Backtest / Algo / Live). Mass Builder Config tab adds same 3 dropdowns; `mass_builder.py:build_strategy_config` threads model kwargs onto each spawned strategy. Admin page renamed "Backtest Models" → "Backtest / Algo Models" reflecting same registry serves both consumers.
- [x] 8.7-stored-trades-staleness. [Investigate] **stored_trades stuck at April dates** — root-caused 2026-05-11. Two issues:
  (a) Phase 40's `hydrate_strategy_trades` was back-filling `strategy['stored_trades']` from the trades table without a data_source filter. Since the trades table only contained algo (cache_locked) data, the divergence endpoint's "backtest" lane was silently reading algo data and returning identical values to the "cache" lane — explaining the fake "perfect alignment" observed all week.
  (b) `recompute_and_persist_stored_trades` was writing the full backtest output to `strategies.stored_trades` JSONB on every refresh. Large JSONB writes (2,500+ trades) were silently failing under Supabase intermittent issues, leaving stored_trades stuck at the last successful refresh date.
  **Both fixed by Phase 41 migration below.**
- [x] 8.7-cron-disable-default. [Required] **Cron paused by default** ✅ shipped 2026-05-11 (commit `0801faa`). After a week of Supabase intermittent outages correlated with cron writes (probably circumstantial; couldn't fully rule out), default flipped from ON to OFF. Re-enable via `ALGO_HISTORY_CRON_ENABLED=true` on Railway worker. Manual update path (bulk page or Strategy Detail buttons) remains canonical.
- [x] 8.7-phase41-backtest-to-trades-table. [Required] **Phase 41 — backtest data migrated to trades table** ✅ shipped 2026-05-11 (commits `8b74ddd` + `5fbfe0d`). Completes the storage unification started in Phase 40 (2026-04-24).
  - **Schema:** unique constraint relaxed to include `data_source` so REST + CACHE rows at identical timestamps coexist. Existing NULL data_source rows tagged as `cache_locked` (preserves cron-written history).
  - **Writers:** `recompute_and_persist_stored_trades` + `append_new_backtest_trades_for_strategy` now write to trades table with `data_source='backtest_<model>'` via `trades_store.replace_trades_for_strategy(filter='backtest_%')`. JSONB write skipped on success; defensive fallback only on trades-table error.
  - **Readers:** divergence endpoint reads both lanes from same trades table with distinct filters (`backtest_%` vs `cache_%`). `hydrate_strategy_trades` + `get_stored_trades` default-filter to `'backtest_%'` preserving the semantic that stored_trades = backtest output.
  - **Backfill:** one-time `_backfill_stored_trades_to_table.py` migrated 17,268 of 17,269 stored_trades JSONB entries → trades table with proper data_source labels (1 lost to transient connection drop on sid 136, regenerated on next refresh).
  - **End-to-end verification on sid 152 (2026-05-11):** 259 matched REST↔CACHE entry events with median=0.0s, p95=0.0s, max=0.0s drift — bit-perfect alignment proves REST and CACHE engines produce identical trade sets on 1Min C-type entries. matched_3way=62 events all three lanes converge.
  - **What this unlocks:** real REST↔CACHE divergence in the Divergence tab. Previously both lanes silently read the same algo rows, producing fake "perfect alignment." Now they're distinct.
- [ ] 8.7-rest-vs-cache-divergence-v2. [Deprecated/Superseded] Original design doc at `docs/Design_REST_vs_CACHE_Divergence_v2.md` recommended JSONB map keyed by model (Option B). Phase 41 implementation went with Option C (single trades table + data_source filter) per Kevin's preference. Doc kept for design history but the actual migration shipped under Option C.

**Tasks (Polish):**
- [ ] 8.7h. [Polish] Cache cleanup job — delete bars older than N days (90 default)
- [ ] 8.7i. [Polish] Cache hit-rate metric in admin UI
- [ ] 8.7j. [Polish] Worker restart gap handling — REST-backfill missed window on startup
- [ ] 8.7-engine-truth-replay. [Polish] Wire `bar_engine_states` into Lab tab replay so the engine-truth-precision view becomes available (today the data is captured but not visualized)
- [ ] 8.7-algo-history-redesign. [Polish] Algo history redesign — stop worker from writing trades table, scheduled refresh, per-strategy `last_backtest_through` checkpoint, append-only refreshes. Documented in plan doc + `docs/Plan_Weekend_2026-05-02.md`.

**Tasks (Deferred):**
- [ ] 8.7k. [Deferred] Strategy historical backfill — REST-fill 90 days for active strategies at cache start
- [ ] 8.7l. [Deferred] Per-tick cache (massive volume; defer to Hi-Fi work)
- [x] 8.7-tv-test. [Done 2026-05-04/05] **TV stability test** ✅ 1Min revises within ~5 min; 10Sec doesn't revise after ~15s. Established the 15-min late-print window as the safe "lock-and-forget" boundary. Informed default flip to `ws_agg_locked`.

**Exit Criteria:**
- ✅ Worker writes bars across primary + secondary TFs during RTH (live_bars accumulating)
- ✅ Worker correctly handles WS rebroadcasts as corrections, not duplicate bars
- ✅ Lab tab Alert Lens shows cache-derived indicators + heatmap (matches what engine sees)
- ✅ Per-alert indicator state captured in `alerts.data.indicator_snapshot`
- ✅ Per-bar engine state captured in `bar_engine_states` (data-only foundation)
- [ ] `load_market_data` reads from cache for cached-period queries (M8.7d)
- [ ] Re-running parity sweep shows fidelity at ~100% on cached-period strategies (M8.7g)
- [ ] Live alerts match fresh local backtest on same bars within $0.0001 (M8.7g)

**Monday RTH validation queue (live data, can't be done Saturday):**
- Mon-1: Run `_validate_live_bars_cache.py --hours-back 1` after 30+ min of trading. Confirm `first_close ≠ close` divergence on 1Min bars (~10-20% expected based on Friday's correction rates).
- Mon-2: `SELECT id, data->'indicator_snapshot' FROM alerts ORDER BY id DESC LIMIT 5;` — confirm M4 snapshots populating.
- Mon-3: `SELECT strategy_id, count(*) FROM bar_engine_states WHERE bar_start > now() - interval '1 hour' GROUP BY strategy_id;` — confirm ~14k rows/day capture rate.
- Mon-4: TV stability test (see deferred tasks above).
- Mon-5: Open Lab tab Replay mode on a strategy with morning data; scrub through and verify smooth indicator/heatmap evolution.

---

### Milestone 9: Scale Infrastructure + Pluggable Architecture
**Priority:** Medium — required before AI agents
**Effort:** 2-3 weeks

Ensure the system can handle thousands of strategies, portfolios, and packs. **Also: convert remaining hardcoded built-in patterns into pluggable modules** so the AI can create new pack types without engine changes.

**Scale tasks:**
- [ ] 9a. Batch backtest optimization: run multiple strategies in parallel
- [ ] 9b. Strategy creation API: create strategy from config without UI (for AI agents)
- [ ] 9c. Portfolio creation API: create portfolio from strategy list without UI
- [ ] 9d. Bulk forward test updates: refresh all strategies in one operation
- [ ] 9e. Performance profiling: identify bottlenecks at 100/1000/10000 strategies
- [ ] 9f. Database indexing: ensure queries scale with strategy count
- [ ] 9g. Worker scaling: Railway auto-scale for alert processing

**Mass Builder scaling tasks (critical — blocks AI agents at scale):**
AI agents will fire off thousands of mass searches. Current `run_mass_search()` is fully
sequential and results persist via JSONB. Infrastructure must handle high concurrent
load and long-running searches before agent rollout.

- [ ] 9l. **Parallelize trigger backtests** — `run_mass_search()` currently runs one backtest at a time. Wrap the inner loop with `ThreadPoolExecutor` (pandas/numpy release the GIL, so threads give near-linear speedup up to 4-8 cores). Target: 4-8x speedup on a Railway dyno. Expect implementation ~1 day.
- [ ] 9m. **Job queue + dedicated worker service** — Redis + RQ or Celery. User submits a search via API → queue picks it up → worker runs asynchronously → result lands in DB → user notified when done. Lets users close the browser / walk away. Also enables AI agents to fire off many searches without blocking each other. Railway supports worker services natively. **Tracked in more detail as item 9ac (Tier 3) below.**
- [ ] 9n. **Dedicated results table** — Currently mass search results persist to `mass_search.config_data` JSONB. For scale, create `mass_search_results` table (one row per result) with indexes on user_id, search_id, kpis.daily_r. Enables fast per-result queries (save, filter, sort) and removes JSONB payload size limit.
- [ ] 9o. **Cost accounting per search** — Track CPU seconds, bars processed, backtests run per search. Feeds into usage-based pricing (see Mass Builder pricing tiers in `Monetization_Model.md`). Important for AI agents — without per-search accounting, an agent could burn thousands of dollars of compute uncontrolled.
- [ ] 9p. **Search quotas and rate limits** — Per-user concurrency caps, per-tier backtest count limits, graceful backpressure when queue is full. Without this, one user (or one runaway agent) can monopolize the worker pool.
- [ ] 9q. **Incremental result streaming** — Currently results only appear after the full search completes. For long searches, stream partial results as each (ticker, TF) group finishes. Users can see early results and cancel if trajectory looks bad.
- [ ] 9r. **Calibrate preview estimator from production telemetry** — `[MASS-CALIBRATION]` logs already emit per-run timing. Aggregate into a `search_telemetry` table, fit regression (bars processed → wall time) per TF/ticker type, feed coefficients back into the frontend estimator. Replaces the hardcoded 350ms / 0.05ms constants with data-driven estimates.
- [x] 9s. **Mass Builder Tier 2 — Resume from checkpoint** ✅ SHIPPED 2026-04-21 (commit `5bad436`). Searches now flush a `checkpoint` payload (completed (symbol, tf) pairs + partial_results + diagnostics) to `mass_searches.config_data.checkpoint` at the end of each `(symbol, tf)` group. Orphaned rows on Mass Results show a **Resume** button that calls `POST /api/mass-builder/resume/{search_id}` — backend relaunches the worker with the saved config + checkpoint, skipping already-completed groups. No DB schema change (piggybacked on existing JSONB). See `docs/Trade_Timestamps_Spec_2026-04-17.md` and `project_session_2026-04-21.md` for design context.
- [ ] 9t. **Mass Builder — Two-phase verification flow** — Kevin's proposed workflow: phase 1 uses the current fast post-filter Mass Builder to surface candidates; phase 2 runs each candidate through `recompute_and_persist_stored_trades` (same path as Strategy Detail "Update All Data") for engine-accurate KPIs before saving. Phase-2 can batch and parallelize. Closes the post-filter-vs-gate divergence ("Preview KPIs" warning ships 2026-04-21 as a stopgap). Effort: 1-2 days. Depends on 9l (parallelize backtests).
- [ ] 9u. **Mass Builder — Concurrency cap on running searches** — Limit to N (≤3) concurrent in-flight searches per user; newer submissions queue. Prevents the GIL-serialization contention (observed 2026-04-21 with 5 concurrent running states on one uvicorn worker). Short-term fix before full 9m job queue. Effort: half-day.
- [ ] 9v. **Incremental "Update New Data" on strategy refresh** — Currently `/api/strategies/{id}/refresh` and the My Strategies / My Portfolios "Update All Data" buttons always re-run a full backtest. For long-running strategies, this reloads the entire history window and re-computes every trade. An incremental variant would: (1) read each strategy's `data_refreshed_at` timestamp, (2) load only the new data window since then (plus indicator warmup), (3) run the engine on that slice, (4) append new trades to the existing `stored_trades`. Works well alongside the existing "Update All Data" which stays for full rebuilds. Surface as a second button next to Update All Data on My Strategies + My Portfolios + Strategy Detail. Effort: 1-2 days. Risk: edge cases around warmup overlap and duplicate-trade detection.
- [ ] 9w. **Portfolio Performance vs Plan — dollar-weighted upgrade** — Phase B (2026-04-21) shipped a first-pass R-multiple port on the Live Dashboard tab that reuses `src/charts/PerformanceVsPlan.tsx`. It assumes roughly uniform `risk_per_trade` across visible strategies in a portfolio. For mixed-risk portfolios (e.g., $500/trade on a high-conviction strategy vs $100/trade on explorers) the plan line is slightly miscalibrated because R-math weights each strategy equally. Upgrade: (1) call `usePortfolioCompute(portfolioId, ['benchmark', 'equity_curve_with_strategies'])` — backend already returns `per_strategy` stats (`avg_r`, `var_r`, `trade_frequency`, `risk_per_trade`); (2) client-side recompute weighted benchmark using only **visible** strategies' per-strategy data (mirrors `compute_portfolio_benchmark` in `src/portfolios.py:1590-1713` but applied to the visibility-filtered subset); (3) fork or extend PerformanceVsPlan.tsx to render dollars instead of R, using the recomputed plan/1SD/2SD arrays directly. Actual line = cumulative dollar P&L of visible strategies' forward trades. Effort: 3-4 hours. Depends on: none. Noted by Kevin 2026-04-21; defer until after Phase C/D of portfolio wire-up.
- [ ] 9x. **API container cold start optimization** — Container takes ~2 minutes to bind port after deploy (observed repeatedly 2026-04-21 during portfolio wire-up). Root cause hypothesis: heavy Python module imports (pandas, numpy, pack_registry scan, etc.) at module load time. Every commit on `dev` triggers a full API rebuild even when the commit is frontend-only, because both services build from repo root. Two fixes worth considering: (a) split Dockerfile builds so frontend-only commits don't rebuild the API image (Railway may need a trigger filter); (b) investigate lazy imports for the heaviest modules. Effort: 2-4 hours investigation, variable fix size. Impact: reduces every deploy's user-visible downtime window from 2 min to ~20-30s.
- [ ] 9y. **Backend endpoint NaN/Inf sanitization audit** — Recurring bug class: computed-KPI endpoints (`/api/strategies`, `/api/portfolios/{id}/trades`, any `/compute` include with KPIs) can propagate `float('nan')` / `float('inf')` through their response dict, which Starlette's `json.dumps` rejects with `"Out of range float values are not JSON compliant: nan"` → 500 error → frontend shows "Failed to load X." Fixed on two endpoints 2026-04-21 (commits 501f28b, 70fdec3) via `_sanitize_for_json` wrap; fix the class by (a) reviewing every endpoint handler that returns computed aggregates + wrapping consistently, or (b) adding a response-level middleware that sanitizes before Starlette's render. Effort: 1-2 hours for a sweep; ongoing vigilance needed until response-level middleware lands.
- [ ] 9z. **Account balance auto-populate on portfolio create** — Backend's `account.balance` field stays at 0 for portfolios that haven't had a manual ledger entry, even when `starting_balance` is set. Frontend workaround shipped 2026-04-21: Account tab + Live Dashboard fall back to `starting_balance + deposits − withdrawals` when `account.balance` is 0. Proper fix: set `account.balance = starting_balance` on portfolio create, and recompute on each deposit/withdrawal mutation. Effort: 1 hour.

**Mass Builder — 3-tier scaling path** (sequencing note added 2026-04-22):

Context: on 2026-04-22 a 10-second TF resume crashed the project-wide Supabase instance. Root cause: checkpoint JSONB payload ballooned (hundreds of combos × `stored_trades` each) and the 60s flushes on `--workers 1` API became expensive enough to block the event loop and overwhelm postgrest. The items below are the scaling path for Mass Builder. **Tiers 1 + 3 are the recommended sequence; Tier 2 is an intermediate half-step that is NOT recommended given the eventual AI-agent workload — it's listed for completeness.**

- [ ] 9aa. **Mass Builder Scale — Tier 1 (slim checkpoint + group-boundary flushes)** — Current checkpoint payload includes `stored_trades` and `equity_curve` for every completed combo. For big-data searches (10s TFs, wide confluence searches) the row grows into MBs and every 60s flush rewrites the whole JSONB blob. Fix: (a) strip `stored_trades` and `equity_curve` from `partial_results` before writing to the checkpoint — on resume we skip completed `(symbol, tf)` groups, so per-trade data isn't needed for resume; we recompute at save time via the existing Strategy Detail "Update All Data" path; (b) flush checkpoint only at `(symbol, tf)` group boundaries, not on a 60s timer (60s buys nothing since recovery granularity was already group-level). Expected impact: checkpoint rows drop from MB-scale to KB-scale, Supabase write pressure from big searches goes away. Effort: 30-45 min. **Carries into Tier 3 unchanged** — queue-worker payloads should also be slim, so this is not throwaway work. Status: deferred until after live-trading validation.
- [ ] 9ab. **Mass Builder Scale — Tier 2 (dedicated worker service, no queue)** — Move `mass_builder` out of the API process into its own Railway service. Own memory budget, own restart policy. The worker reads `mass_searches` rows with `status='queued'` directly (polling pattern, same as the existing Ralph Worker service). API becomes a thin client that inserts the row. Doesn't require Redis; uses Supabase as the coordination surface. **NOT RECOMMENDED as a stopping point** for Kevin's AI-agent vision — it's a half-step on the way to Tier 3 and would be ripped out when multi-worker concurrency matters. Listed for completeness in case Tier 3 is slow to land. Effort: ~1 day. If built, rips out cleanly when Tier 3 arrives.
- [ ] 9ac. **Mass Builder Scale — Tier 3 (full job queue)** — Supersedes the in-process worker entirely. Redis + RQ (or Celery) + dedicated worker service(s) on Railway. Concurrency controllable (N workers, per-user caps). Proper retries, priorities, rate limiting. Survives any service restart because the queue is external. Required for Kevin's roadmap vision where AI agents fire off many searches in parallel as their own users. Redis adds ~$5-10/mo infra cost on Railway. This is the correct destination — **recommend jumping straight to Tier 3 after Tier 1 lands**, skipping Tier 2. Effort: 2-3 days. Consolidates / supersedes items 9m, 9u (concurrency cap becomes a queue rate limit), and makes 9p (quotas) a natural extension.

**Portfolio Trade History — live-trading verification polish:**

Context: on 2026-04-22, trade-history merging of live alerts + backtest shipped (commit `a75396e`). Frontend now shows four status badges (Backtest / Forward Test / Matched / Live / Phantom). Next ask from Kevin: make each row inspectable so he can verify the webhook actually fired and carried the right payload before taking real trades.

- [ ] 9ad. **Portfolio Trade History — Details drawer per row** — Add a trailing "Details" column with a button on every row in the Trade History table. Clicking opens a drawer / modal showing: (a) the full trade record (entry/exit prices, slippage-R, risk-per-trade, quantity planned vs executed, exec_type); (b) **the webhook payload that was dispatched** — rendered template body post-`{{placeholder}}` substitution, the webhook URL target, the HTTP method, headers; (c) **delivery status** — HTTP response code, response body excerpt, timestamp, retry count, final outcome (delivered / failed / pending). Data source: join on `webhook_deliveries` (or equivalent from `src/alerts.py` delivery log) keyed by `alert_id`. Purpose: pre-live-trading verification — Kevin needs "yes, this fired, here's proof" before pointing it at real money. Effort: 2-3 hours (backend: surface delivery log per alert in `/portfolios/{id}/trades` or a companion endpoint; frontend: drawer/modal component + wiring). Depends on: webhook delivery log persistence (already exists via commit history around Phase 39 webhook redesign). This is the **highest priority portfolio-polish item** for going live.
- [ ] 9ae. **Portfolio Trade History — Embed Trade Replay modal** — The Strategy Builder Enhanced Trade History has a "Trade Number Replay Modal" that replays a single trade bar-by-bar (commit history under Strategy Builder polish). Embed the same component on Portfolio Trade History rows so clicking the trade-number cell opens the replay in-context. Useful for verifying entry/exit conditions match the engine's expectations when a Live or Matched alert looks off. Effort: 1-2 hours (the modal already exists — work is extraction into a shared component + wiring the portfolio row's trade number to open it with the correct strategy_id + trade_index). Lower priority than 9ad; defer until Kevin confirms 9ad is working and webhook verification flows are smooth.
- [ ] 9al. **Portfolio Account tab — wire Change History end-to-end** — Account tab's "Change History" card is currently a stub (`PortfolioDetailPage.tsx` ~line 2130 renders hard-coded "No change history yet"). Legacy Streamlit has `portfolios.add_change_log_entry(portfolio, change_type, details, description)` that records `'strategy_added' / 'strategy_removed' / 'risk_adjusted' / 'requirement_set_changed'` events into `portfolio.change_log`, but the new React frontend never got (a) an endpoint to read it, (b) a hook/component to render it, or (c) writes from the mutation paths (ledger add, risk edit, req-set swap, strategy add/remove). Build: `GET /api/portfolios/{id}/change-log` router, `usePortfolioChangeLog` hook, a `<ChangeHistoryList>` component, plus call sites in `add_ledger_entry` / portfolio PUT / strategy-allocation mutations. Effort: ~half day. Discovered 2026-04-23 while diagnosing "added a ledger entry but change history stays empty" — user confirmed the stub is expected and parked this as a dedicated item.
- [ ] 9am. **Surface subscribe-state clearly in UI — "strategy in portfolio" vs "portfolio → webhook group"** — The two "subscription" concepts get confused in casual talk: (1) strategy-allocation (strategy appears on portfolio's Strategies tab) = trades show up in that portfolio's views; (2) webhook-group-assignment (portfolio.webhook_group_id) = where that portfolio's dispatches actually POST. Having a strategy in a portfolio does NOT mean webhooks will fire for it — the portfolio also needs a webhook group. Proposed UX: (a) Strategy Detail page — panel listing "In portfolios: P19→TTP ✓ / P27→(no group)" so user sees the full dispatch fan-out; (b) Portfolio's Strategies tab — per-row badge "🔔 via {group_name}" vs "🔕 no webhook group"; (c) Portfolio's Webhooks tab — clarify copy that the group subscription is independent of which strategies are allocated. Effort: ~half day. Discovered 2026-04-23 — Kevin confused why one portfolio dispatched and another didn't when both had the same strategy listed, root cause was different webhook_group_id states.
- [ ] 9an. **Supabase statement-timeout relief** — On 2026-04-23 Supabase went "unhealthy" 4-5 times, each triggering `statement timeout` / `read operation timed out` errors on alert saves, algo-trade persists, and monitor_status updates. Required manual project restart each time. Contributing factors to investigate: (a) per-bar alert writes bunching under high TF count; (b) Mass Builder checkpoint JSONB flushes (slim checkpoint 9aa directly reduces this); (c) realtime broadcast volume (post-738ab38 throttle is 1/sec/user but worker-side broadcasts for every watched symbol/TF regardless of who's watching — 9ai); (d) `algo_trade_persist` writes a full trade record every exit — large payloads × many strategies. **Phase 40 trades-table migration (2026-04-24) directly addressed (d) — `algo_trade_persist` is now a single-row INSERT.** First pass on remaining contributors: instrument which call is slowest under load, then attack biggest contributor. Temporary workaround: restart Supabase project when it wedges.
- [x] 9ao. **Phase 40 — Normalize trades out of `strategies.stored_trades` JSONB into a `trades` table.** SHIPPED 2026-04-24. JSONB column grew to 2.5 MB on strategy 114 (~3000 trades) and 3+ MB on strategy 71 (~19,578 trades); every exit rewrote the entire column via `_persist_algo_trade`, causing statement-timeout cascades. Migration (`src/migrations/phase40_trades_table.sql`): new `trades` table with 14 hot columns + `data` JSONB + RLS + 6 indexes; one-shot backfill via `jsonb_array_elements`. Code (`src/trades_store.py` + `db.py` + `worker.py`): feature-flagged via `USE_TRADES_TABLE` env var. When ON, writes go to trades table as single-row INSERTs; reads hydrate via `_row_to_strategy` so frontend DTO shape is preserved. PostgREST/Supabase enforces an undocumented 1000-row cap on responses — `load_trades_admin` paginates via `.range()`. Remaining cleanup: remove flag branches (Commit 4), eventually drop `stored_trades` column (weeks later). See `project_session_2026-04-24.md` for full detail.
- [ ] 9ap. **Paginated `/api/strategies/{id}/trades` endpoint** — 9ao hydrates the full trade list into `strategy.stored_trades` on every detail load, which is ~20 paginated Supabase calls for strategy 71 (19k trades). Fine for now; frontend renders OK. Future: add `?page=N&page_size=100` pagination; have the frontend lazy-load trade rows on scroll. Scope: ~half day. Land with Commit 4 cleanup.
- [ ] 9aq. **Persist `_position_quantity` tracker across Worker restarts** — pre-existing exit-webhook-quantity-0 bug resurfaces every Worker restart (seen twice during 2026-04-24 flag-flip testing). In-memory tracker loses "how many shares did entry fill for strategy X, portfolio Y" when Worker process restarts; any exit for a position that opened pre-restart sends `quantity: 0` and SignalStack rejects with 400. Fix: snapshot `_position_quantity` into `monitor_status.engine_state` alongside position state, restore on worker `_run` startup via existing `_seed_deployed_capital` path. Scope: 1-2 hours. Currently mitigated by only flipping env vars during flat-position windows.
- [ ] 9ar. **Admin `forward_test_start` editor** — SHIPPED 2026-04-24 (commit `dc0fce0`). Inline date-picker on Strategy Detail → Configuration tab → Strategy Setup card. Unblocks the "recreate old strategy under current schema + restore original start date" workflow. Backend endpoint `PATCH /api/strategies/{id}/forward-test-start`. Does NOT trigger recompute — caller runs Refresh Data separately if needed. (Tracked as done; keeping in roadmap for discoverability.)
- [ ] 9as. **Harden `_strategy_to_row` against partial-update config wipes** — Classic feedback_jsonb_partial_updates bug has now bit THREE times (2026-04-14 strategies 70+71; 2026-04-24 strategies 129+130 via my own set_forward_test_start endpoint). Commit `8f66538` fixed `update_strategy_db` at the caller level, but the helper itself still emits `config: {}` unconditionally. Proper fix: add a `partial=True` flag to `_strategy_to_row` — when true, omit `config` from output if no config-bucket fields were in the input. Every caller gets safety by default. Scope: 30 min + refactor every caller to pass `partial=True` for update paths. Eliminates the bug class permanently.
- [ ] 9at. **Drop `alert_tracking_enabled` column entirely** — Column was dead weight by 2026-04-24. Flag was never checked in the worker's alert-save path (audit confirmed via grep of _fire_alert / dispatch). Toggle UI added then removed same session. All existing strategies bulk-UPDATE'd to true. Column still exists as legacy; dashboard summary still uses it for "monitored_count". Future: drop the column after updating dashboard logic to count on `entry_trigger_confluence_id IS NOT NULL` or similar. Low priority; harmless as-is.
- [ ] 9au. **Backtest ↔ Live Parity Simulator** — HIGH PRIORITY 2026-04-24. Audit found 9 strategies producing forward algo trades but zero live alerts (strategies 50/51/59/63/64/66/111/122/129). Likely cause: indicator/trigger code drifts between unified_engine batch path (used by backtest + recompute) and ralph_engine incremental path (used by live worker). State-machine triggers like `_detected` suffix particularly suspicious — depend on previous-bar state which live mode may not initialize correctly. The Parity Simulator scaffold already exists in UI (UserPacksPage.tsx:1465 + PackBuilderPage.tsx:880) but body is `{{ai_response}}` placeholder. Build the engine: replay historical bars one-at-a-time through ralph_engine's IncrementalIndicatorEngine + TriggerEvaluator (no live data needed — simulates production WebSocket arrival), capture trigger fires; compare against unified_engine's batch-path fires; surface drift visually. V1 = single-pack, single-TF, default config. V2 = strategy-detail mode (composes multiple packs), cross-TF, mid-stream-restart simulation (catches state-machine bugs). Effort: ~1 day for V1, ~half day extra for strategy-detail mode. Plan: `docs/Parity_Simulator_Plan_2026-04-24.md`. Backup: branch `dev-backup-pre-parity-simulator-2026-04-24` at `be78ad3`. **Without this, every new confluence pack Kevin creates is a coin flip on whether it'll fire alerts live — paralyzing pack development.**

**Worker scale path — from dozens to thousands of strategies** (captured 2026-04-22 after Worker OOM'd on 74 strategies mid-session):

Kevin's vision: thousands of strategies running simultaneously across many portfolios, with AI agents managing their own strategy rosters as distinct users. Current architecture (single Railway Worker, in-process RalphEngine) has already hit its ceiling at 74 strategies. Items below are the scaling path. Most are independent and can be sequenced by impact once we pick an order.

- [ ] 9af. **Bump Worker Railway memory tier** — *NOT A CODE CHANGE.* Hobby-plan Worker has ~512MB; one paid tier up is 1-4GB for ~$5-20/mo. Single highest-leverage fix for "run 100+ strategies reliably" before we invest in engine surgery. Kevin's call when to pull this lever. No engineering effort; just a Railway dashboard change. Recommended as the FIRST action if Worker OOMs recur after thread-lock + cache fixes (e1cd607).
- [ ] 9ag. **Throttle monitor_status writes** — Worker currently POSTs `monitor_status` upsert ~every 2 seconds while running. With N users × N writes/user = constant Supabase write pressure (contributes to the recurring "Supabase unhealthy" pattern). Bump interval to 30s (already the `HEARTBEAT_INTERVAL` constant; actual emission is somewhere else and doesn't respect it). Effort: 30 min to find + fix. Impact: ~15× reduction in status-write pressure, negligible UI lag.
- [ ] 9ah. **Worker memory profile + engine-state trim** — Ralph holds per-strategy: indicator histories (unbounded deques?), shadow engines for secondary TFs (one per cross-TF confluence pair), position state machine state, MTF confluence buffer. Growth pattern: scales with strategy count × TF count × market-hours bars. Tasks: (a) profile a running worker with `tracemalloc` snapshots 1h apart to locate the largest-growing objects; (b) cap bar-history deques to the minimum required for indicator warmup (most indicators only need their own lookback, not the full day); (c) identify whether shadow engines can share computation across strategies that use the same (symbol, TF) pair. Effort: 1-2 days investigation + variable fix size. Impact: 3-5× reduction in per-strategy memory plausible.
- [ ] 9ai. **Presence-aware live chart broadcasts** — Kevin's insight 2026-04-22: the worker broadcasts bar updates for every active (symbol, TF) even when no UI is watching. Current state (post-738ab38): throttled to 1/sec per user total — blunt but limits total rate. Smarter: track which (symbol, TF) the frontend is actively viewing via Supabase realtime presence, and broadcast only for those. Alerts + engine computation are unaffected (those run regardless — required for trade generation). Effort: 1 day (frontend presence heartbeat + worker-side subscription filter). Impact: reduces Supabase realtime load proportionally to "watched vs total" — at thousands of strategies, this goes from "impossible" to "trivial" because only a handful are ever being watched. Prerequisite for chart-scalability at 100+ strategies.
- [ ] 9aj. **Multiple Worker instances — horizontal partitioning** — Route user/symbol traffic across N Worker replicas. Each Worker handles a bounded subset of strategies, keeping per-process memory predictable. Partition options: (a) by user_id — natural boundary, each user's engine stays in one Worker; (b) by symbol — all strategies on SPY run on Worker-1, TSLA on Worker-2, etc. Option (a) simpler; option (b) better for symbol-dedup savings. Requires: shared coordination (which Worker owns which subset) via monitor_status table + heartbeat. Railway supports multiple services natively. Effort: 2-3 days. Unblocks "hundreds reliably" once 9af + 9ah + 9ai land.
- [ ] 9ak. **Event-sourced architecture + externalized engine state** — Long-term destination. Worker becomes a stateless consumer of bar event streams; per-strategy state (position, indicator history, MTF buffer) lives in Redis. Crash-recoverable without seeding. Horizontally scalable without sticky-session complexity (any Worker can handle any alert). Aligns with Mass Builder Tier 3 (9ac) — same Redis + queue + stateless-worker pattern. Building both at once means one infra investment unlocks "thousands of strategies + thousands of searches." Effort: 5-7 days. Do when 9aj's partitioning pattern starts showing its own limits (~500 strategies per user).

See also: the existing items 9m (job queue — now unified under 9ac), 9n (dedicated results table — complements any tier but especially Tier 3), 9o (cost accounting — unlocked by Tier 3 telemetry), 9p (search quotas — natural once a queue exists).

**Pluggable architecture tasks (eliminate hardcoding):**
- [ ] 9h. **Pluggable stop loss / take profit packs** — Currently `unified_engine.py` has hardcoded `if/elif method == 'atr' / 'fixed_dollar' / 'percentage' / 'swing' / 'risk_reward'` branches (~lines 1995-2070). Convert into a registry pattern (similar to execution types). Each stop/target method becomes a module: `StopLossPack` base class with `compute_initial_stop()`, `compute_trail()`, `should_move_to_breakeven()` methods. Risk management packs (currently in `risk_management_packs.py`) become installable like indicator packs. New stop types (e.g., "VWAP-anchored swing stop", "psychological round number stop") added without engine changes.
- [ ] 9i. **Convert built-in indicators to user pack format** — Currently built-in indicators (EMA Stack, MACD, VWAP, RVOL, UTBOT, etc.) are inline incrementally computed in `unified_engine.py`. They have hardcoded gate maps in `_compute_ib_gates`, `_get_ib_checks`, `_update_cached_levels`. Convert each built-in into a pre-installed user pack with manifest + indicator.py + interpreter.py. The fallback paths added during M5 already handle user pack triggers correctly — once built-ins are converted, the hardcoded paths can be deleted.
- [ ] 9j. **Remove hardcoded template sets** — Once built-ins are user packs, delete `OVERLAY_TEMPLATES` and `OSCILLATOR_TEMPLATES` sets from `classify_and_serialize_chart_data()`. Classification becomes purely `display_type`-driven.
- [ ] 9k. **Document the plugin contract** — A clear spec for "what makes a pluggable thing": indicator packs, execution type packs, stop loss packs, take profit packs, general packs. All follow the same install → register → use pattern.

**Why this matters:** Currently when we hit a built-in code path that doesn't handle user packs, we add a fallback. The fallback list has grown but is mostly complete. Going pluggable removes the hardcoded path entirely, so there's no fallback to maintain. New pack types (e.g., "Order Block Detection", "Volume Profile") become first-class without any engine changes.

**Hardcoded paths inventory (for reference during 9h-9k):**
- `unified_engine.py:73-90` — `INTRABAR_LEVEL_MAP` initial built-in entries
- `unified_engine.py:96-104` — `_IB_L_TYPE_TRIGGERS` initial set
- `unified_engine.py:1141-1158` — `_update_cached_levels` hardcoded built-in column names
- `unified_engine.py:1095-1118` — `_get_ib_checks` hardcoded `IB_MAP`
- `unified_engine.py:1166-1205` — `_compute_ib_gates` hardcoded `gate_map`
- `unified_engine.py:1995-2070` — Stop loss / take profit method `if/elif` branches
- `unified_engine.py:756-985` — `evaluate_bar_close` inline interpreter logic for built-in packs
- `backtest_service.py:181-183` — `OVERLAY_TEMPLATES` / `OSCILLATOR_TEMPLATES` hardcoded sets
- `backtest_service.py:215-247` — Main classification loop iterating `get_enabled_groups()` (works for user packs but special-cases EMA period resolution)
- `unified_engine.py:2340-2358` — `_BUILTIN_INTERPS` set for excluding built-ins from user pack collection

**Exit Criteria:** System handles 1000+ strategies per user without degradation. API endpoints exist for programmatic strategy/portfolio creation. Built-in indicators and stop/target methods are converted to pluggable packs with no hardcoded fallback paths in the engine.

---

### Milestone 10: AI Agent Integration
**Priority:** Future — after Milestones 1-9 are stable
**Effort:** 2-4 weeks

Enable AI agents to autonomously create packs, test strategies, and surface the best trading opportunities.

**Tasks:**
- [ ] 9a. Agent API: create pack → install → run backtests → evaluate → save or discard
- [ ] 9b. Strategy scoring system: rank strategies by risk-adjusted return, consistency, edge quality
- [ ] 9c. Agent orchestration: manage multiple agents running concurrently
- [ ] 9d. Results dashboard: surface top-performing strategies/packs found by agents
- [ ] 9e. Cost management: rate-limit AI API calls, track token usage per agent
- [ ] 9f. Human review workflow: agent proposes → human approves → strategy goes live

**Exit Criteria:** AI agent can autonomously create 100 packs, generate 1000 strategy variations, and surface the top 10 by risk-adjusted return — all without human intervention.

---

### Milestone 10.5: Pack Polish — Directional Classification
**Priority:** Low — quality-of-life, defer until before marketplace work
**Effort:** ~2-4 hours

Add optional BULL/BEAR/NEUTRAL classification to user pack output states so newer
packs (rsi_zones, supertrend, swing_123, etc.) display consistent directional
grouping in the Mass Builder TF Confluence selector. Built-in templates already
classify via `INTERPRETER_DIRECTION_MAP` (added in M7); user packs do not.

**Critical design constraint:** classification is **display-only**. It must
never gate trigger selection in the Strategy Builder or any other context. If
a state is labeled BEAR, the user can still use it in LONG strategies.

**Tasks:**
- [ ] 10.5a. Extend manifest spec: optional `output_directions: Dict[state, "BULL"|"BEAR"|"NEUTRAL"]`
- [ ] 10.5b. Merge user pack `output_directions` into `INTERPRETER_DIRECTION_MAP` at load time
- [ ] 10.5c. Pack Builder UI: add dropdown per output state on the Outputs config step
- [ ] 10.5d. Update AI pack generator prompt to suggest directional classification
- [ ] 10.5e. Backfill existing user pack manifests (rsi_zones, supertrend, swing_123, sr_channels, bollinger_bands, stochastic_oscillator)
- [ ] 10.5f. Add docs note clarifying display-only semantics

**Exit Criteria:** All user packs show BULL/BEAR/NEUTRAL groupings in the Mass Builder. Strategy Builder trigger lists remain unfiltered by direction.

---

### Milestone 11: Marketplace Foundation
**Priority:** Future — after live trading is proven
**Effort:** 4-8 weeks

Enable users to share and monetize packs and strategies.

**Tasks:**
- [ ] 7a. Public pack discovery: browse, search, filter by category/performance
- [ ] 7b. Pack licensing: free, paid, subscription tiers
- [ ] 7c. Revenue sharing: creator gets percentage of subscription revenue
- [ ] 7d. Pack reviews and ratings
- [ ] 7e. Strategy templates: pre-built strategies using public packs
- [ ] 7f. Creator dashboard: earnings, downloads, usage analytics

**Exit Criteria:** Users can publish packs, other users can discover and use them, creators receive revenue share.

---

## Document Cleanup

The following documents are **superseded by this roadmap** and can be archived:
- `ACTIVE_WORK.md` — Outdated 6-phase plan from earlier sessions
- `HANDOFF_VSCODE.md` — One-time handoff doc, context already absorbed
- `HANDOFF.md` — Earlier handoff, fully superseded

The following documents remain **active and relevant:**
- `CLAUDE.md` — Charting & data conventions (still accurate)
- `docs/RoR_Trader_PRD.md` — Product requirements (reference)
- `docs/Webhook_Event_System.md` — Webhook design (reference for Milestone 3)
- `docs/Execution_and_Fidelity_Playbook.md` — Execution type reference
- `docs/Monetization_Model.md` — Business model (reference for Milestone 7)
- This document (`docs/Roadmap_To_Scale.md`) — Active working roadmap

---

## Key Architecture Decisions

1. **One engine for all trade generation.** The unified engine handles both built-in and user packs. Built-in packs use incremental computation (O(1)/bar). User packs use pre-computed DataFrame columns from the batch pipeline. Both flow through the same position state machine, execution logic, and trade recording.

2. **Execution types are pluggable modules.** Each execution type (C, L, LC, CC, and future types) is a class implementing a standard interface. The engine doesn't know about specific execution types — it delegates to the registered module. New execution types require no engine changes.

3. **Pack Builder produces complete, validated packs.** AI generates the code, validation catches issues before install, Sandbox verifies behavior. The human approves the pack before it goes live. This flow must be reliable enough for AI agents to use autonomously.

4. **Ralph engine shares execution modules with unified engine.** The live alert engine uses the same execution type classes as the backtest engine. This guarantees that live alerts fire at the same times as backtest signals.

5. **Scale through parallel workers, not engine optimization.** Instead of making one engine process 10,000 strategies, run 100 workers each processing 100 strategies. Railway handles auto-scaling. The engine stays simple and correct.
