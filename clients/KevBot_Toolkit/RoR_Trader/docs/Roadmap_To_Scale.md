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

### Milestone 4: Pack Builder Polish
**Priority:** Medium — reliability of pack creation
**Effort:** 3-5 days

Ensure packs created through the Pack Builder are consistently reliable and work seamlessly across all features.

**Tasks:**
- [ ] 4a. [Required] Fix remaining validation gaps (from testing — add checks as discovered)
- [ ] 4b. [Required] Update `pack_builder_context.md` with golden child reference (EMA PP V2 as template)
- [ ] 4c. [Required] Add Swing 1-2-3 as Pack Builder test case — validate CC triggers work end-to-end
- [ ] 4d. [Polish] Pack status workflow: Verification → Private → Public (persist status, gate Strategy Builder visibility)
- [ ] 4e. [Polish] Pack versioning: installed pack is immutable, new versions create new packs
- [ ] 4f. [Polish] Delete protection: warn if strategies reference the pack
- [ ] 4g. [Deferred] Legacy pack cleanup: mark old TF Confluence packs as "Legacy (Default)"
- [ ] 4h. [Required] **Chart Preview tab** — Wire state-colored background + trigger markers (like Streamlit). Remove duplicate chart module. Show conditions as background color changes, triggers as markers on candles.
- [ ] 4i. [Polish] **Signal Validation tab** — Wire real data: run indicator on sample data, count trigger fires, verify all states reached, per-trigger breakdown
- [ ] 4j. [Deferred] **Parity Simulator tab** — Nice-to-have. Backtest ↔ live engine parity comparison. May be complex; evaluate feasibility.
- [ ] 4k. [Required] Create several user packs through the full Pack Builder flow and verify each one end-to-end
- [ ] 4l. [Polish] **Sandbox: PB/CB fidelity in confluence dropdown** — When Hi-Fi enabled, show [PB] and [CB] variants of each confluence condition. Display fidelity type in selector labels and heatmap legends.
- [ ] 4m. [Polish] **Sandbox: Confluence heatmap on trade drill-down** — Show PB/CB heatmap pane on the 1-second TradeZoomModal, matching how it appears in the Strategy Builder drill-down.
- [ ] 4n. [Polish] **Sandbox: Move Hi-Fi toggle before confluence** — Hi-Fi toggle should precede confluence selector so the user enables it first, then sees PB/CB variants appear in the dropdown.
- [ ] 4o. [Polish] **Bar count exit variations** — Read N from the confluence group's parameters instead of hardcoding 4. Different bar count exit variations (N=3, 4, 5, etc.) should pass the correct N to the engine.
- [ ] 4p. [Deferred] **Unify User Packs into TF Confluence page** — User packs are just TF confluence packs with a different author. Show all packs on one page with filter (Built-in / My Packs / Public) instead of separate User Packs page. Pack Builder creates packs of the appropriate type. Custom packs get a "My Pack" badge.

**Exit Criteria:** User can create 10 different packs through the Pack Builder and every one works correctly in Strategy Builder, Sandbox, and live alerts without manual intervention.

---

### Milestone 5: Strategy Builder & Strategy Detail Polish
**Priority:** High — strategies are the core product
**Effort:** 1-2 weeks

Ensure the Strategy Builder and Strategy Detail pages work flawlessly with both built-in and user packs. Fix any remaining bugs, polish the UX, and verify end-to-end consistency.

**Tasks:**
- [ ] 5a. Verify Strategy Builder works with user pack triggers (entry + exit)
- [ ] 5b. Verify bar count exit, stop loss packs, take profit packs all work correctly with user packs
- [ ] 5c. Verify Strategy Detail page renders correctly for strategies using user packs (chart, KPIs, equity curve, trade history)
- [ ] 5d. Fix any chart/indicator display issues for user pack strategies
- [ ] 5e. Verify forward testing works with user packs
- [ ] 5f. Verify alert tracking works with user packs
- [ ] 5g. Fix any remaining Strategy Detail bugs surfaced during testing
- [ ] 5h. Polish Strategy Builder UX: ensure trigger dropdowns, analysis tabs, and backtest flow are smooth
- [ ] 5i. Create several real strategies using user packs — QA the full flow end-to-end

**Exit Criteria:** User can create a strategy using any combination of built-in and user packs, run backtests, view detailed results, enable forward testing, and enable alert tracking — all working correctly.

---

### Milestone 6: Mass Strategy Builder
**Priority:** High — required for scale testing
**Effort:** 1-2 weeks

Wire the Mass Strategy Builder to work with the unified engine and user packs. This is the tool that enables testing hundreds of strategy variations in one batch.

**Tasks:**
- [ ] 6a. Verify Mass Builder works with unified engine (not legacy `generate_trades()`)
- [ ] 6b. Wire user pack triggers into Mass Builder trigger selection
- [ ] 6c. Verify Mass Builder results are consistent with individual Strategy Builder backtests
- [ ] 6d. Wire Hi-Fi mode into Mass Builder
- [ ] 6e. Bulk save: save top N strategies from Mass Builder results
- [ ] 6f. Progress tracking: show real-time progress during mass backtest runs
- [ ] 6g. Performance optimization: parallelize mass backtests where possible

**Exit Criteria:** User can run 100+ strategy variations through Mass Builder, results match individual backtests, top strategies can be saved in bulk.

---

### Milestone 7: Portfolios Polish
**Priority:** High — portfolios combine strategies into tradeable units
**Effort:** 1-2 weeks

Polish the portfolio system to work reliably with user pack strategies. Verify aggregation, risk management, and visualization.

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

### Milestone 8: Scale Infrastructure
**Priority:** Medium — required before AI agents
**Effort:** 1-2 weeks

Ensure the system can handle thousands of strategies, portfolios, and packs.

**Tasks:**
- [ ] 8a. Batch backtest optimization: run multiple strategies in parallel
- [ ] 8b. Strategy creation API: create strategy from config without UI (for AI agents)
- [ ] 8c. Portfolio creation API: create portfolio from strategy list without UI
- [ ] 8d. Bulk forward test updates: refresh all strategies in one operation
- [ ] 8e. Performance profiling: identify bottlenecks at 100/1000/10000 strategies
- [ ] 8f. Database indexing: ensure queries scale with strategy count
- [ ] 8g. Worker scaling: Railway auto-scale for alert processing

**Exit Criteria:** System handles 1000+ strategies per user without degradation. API endpoints exist for programmatic strategy/portfolio creation.

---

### Milestone 9: AI Agent Integration
**Priority:** Future — after Milestones 1-8 are stable
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

### Milestone 10: Marketplace Foundation
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
