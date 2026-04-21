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
**Status:** In progress (2026-04-14). M8 WIP committed + pushed (`00f2450`). **Revised priority:** move to M8.5 (Live Data) next after validating PortfolioNewPage end-to-end with a user-pack strategy. Verification tasks (7c/7d/7e/7f/7g) deferred until live data is available — they'll be easier to exercise with real-time signals flowing.

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
- [ ] 9m. **Job queue + dedicated worker service** — Redis + RQ or Celery. User submits a search via API → queue picks it up → worker runs asynchronously → result lands in DB → user notified when done. Lets users close the browser / walk away. Also enables AI agents to fire off many searches without blocking each other. Railway supports worker services natively.
- [ ] 9n. **Dedicated results table** — Currently mass search results persist to `mass_search.config_data` JSONB. For scale, create `mass_search_results` table (one row per result) with indexes on user_id, search_id, kpis.daily_r. Enables fast per-result queries (save, filter, sort) and removes JSONB payload size limit.
- [ ] 9o. **Cost accounting per search** — Track CPU seconds, bars processed, backtests run per search. Feeds into usage-based pricing (see Mass Builder pricing tiers in `Monetization_Model.md`). Important for AI agents — without per-search accounting, an agent could burn thousands of dollars of compute uncontrolled.
- [ ] 9p. **Search quotas and rate limits** — Per-user concurrency caps, per-tier backtest count limits, graceful backpressure when queue is full. Without this, one user (or one runaway agent) can monopolize the worker pool.
- [ ] 9q. **Incremental result streaming** — Currently results only appear after the full search completes. For long searches, stream partial results as each (ticker, TF) group finishes. Users can see early results and cancel if trajectory looks bad.
- [ ] 9r. **Calibrate preview estimator from production telemetry** — `[MASS-CALIBRATION]` logs already emit per-run timing. Aggregate into a `search_telemetry` table, fit regression (bars processed → wall time) per TF/ticker type, feed coefficients back into the frontend estimator. Replaces the hardcoded 350ms / 0.05ms constants with data-driven estimates.
- [ ] 9s. **Mass Builder Tier 2 — Resume from checkpoint** — Searches persist per-group progress (completed_groups, partial_results, last_step) to DB at each (ticker, tf) boundary. On orphan detection (Tier 1 already ships startup cleanup, 2026-04-21), add a "Resume" button that restarts the worker skipping already-completed groups. Preserves multi-hour research from deploy-kills. Effort: 1/2–1 day. Precedes 9m (job queue) since resumable in-process searches mitigate most of the pain without new infrastructure.
- [ ] 9t. **Mass Builder — Two-phase verification flow** — Kevin's proposed workflow: phase 1 uses the current fast post-filter Mass Builder to surface candidates; phase 2 runs each candidate through `recompute_and_persist_stored_trades` (same path as Strategy Detail "Update All Data") for engine-accurate KPIs before saving. Phase-2 can batch and parallelize. Closes the post-filter-vs-gate divergence ("Preview KPIs" warning ships 2026-04-21 as a stopgap). Effort: 1-2 days. Depends on 9l (parallelize backtests).
- [ ] 9u. **Mass Builder — Concurrency cap on running searches** — Limit to N (≤3) concurrent in-flight searches per user; newer submissions queue. Prevents the GIL-serialization contention (observed 2026-04-21 with 5 concurrent running states on one uvicorn worker). Short-term fix before full 9m job queue. Effort: half-day.
- [ ] 9v. **Incremental "Update New Data" on strategy refresh** — Currently `/api/strategies/{id}/refresh` and the My Strategies / My Portfolios "Update All Data" buttons always re-run a full backtest. For long-running strategies, this reloads the entire history window and re-computes every trade. An incremental variant would: (1) read each strategy's `data_refreshed_at` timestamp, (2) load only the new data window since then (plus indicator warmup), (3) run the engine on that slice, (4) append new trades to the existing `stored_trades`. Works well alongside the existing "Update All Data" which stays for full rebuilds. Surface as a second button next to Update All Data on My Strategies + My Portfolios + Strategy Detail. Effort: 1-2 days. Risk: edge cases around warmup overlap and duplicate-trade detection.
- [ ] 9w. **Portfolio Performance vs Plan — dollar-weighted upgrade** — Phase B (2026-04-21) shipped a first-pass R-multiple port on the Live Dashboard tab that reuses `src/charts/PerformanceVsPlan.tsx`. It assumes roughly uniform `risk_per_trade` across visible strategies in a portfolio. For mixed-risk portfolios (e.g., $500/trade on a high-conviction strategy vs $100/trade on explorers) the plan line is slightly miscalibrated because R-math weights each strategy equally. Upgrade: (1) call `usePortfolioCompute(portfolioId, ['benchmark', 'equity_curve_with_strategies'])` — backend already returns `per_strategy` stats (`avg_r`, `var_r`, `trade_frequency`, `risk_per_trade`); (2) client-side recompute weighted benchmark using only **visible** strategies' per-strategy data (mirrors `compute_portfolio_benchmark` in `src/portfolios.py:1590-1713` but applied to the visibility-filtered subset); (3) fork or extend PerformanceVsPlan.tsx to render dollars instead of R, using the recomputed plan/1SD/2SD arrays directly. Actual line = cumulative dollar P&L of visible strategies' forward trades. Effort: 3-4 hours. Depends on: none. Noted by Kevin 2026-04-21; defer until after Phase C/D of portfolio wire-up.

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
