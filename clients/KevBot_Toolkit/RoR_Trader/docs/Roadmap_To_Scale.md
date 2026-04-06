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

### Milestone 1: Engine Parity (Option A)
**Priority:** Critical — blocks everything else
**Effort:** 1-2 days

Make user packs flow through the unified engine by reading pre-computed DataFrame columns. Eliminate the `generate_trades()` fallback entirely.

**Tasks:**
- [ ] 1a. Modify `TriggerEvaluator.evaluate_bar_close()` to check for pre-computed interpreter columns in the DataFrame when the interpreter key is not in `self.required_interpreters`
- [ ] 1b. Modify `TriggerEvaluator` to check for pre-computed trigger columns (`trig_*`) in the DataFrame when the trigger key is not in `self.required_triggers`
- [ ] 1c. Remove the `_use_batch` / `generate_trades()` fallback from `backtest_service.py`
- [ ] 1d. Add `bar_count_exit` control to SandboxPanel
- [ ] 1e. Verify: RSI Zones pack produces identical trades through unified engine as through batch engine
- [ ] 1f. Verify: built-in packs (EMA PP V2, UT Bot) still produce correct trades (no regression)

**Exit Criteria:** All packs (built-in and user) flow through the unified engine. Sandbox matches Strategy Builder exactly.

---

### Milestone 2: Execution Type Extraction
**Priority:** High — enables modular execution types, testable in isolation
**Effort:** 1-2 weeks

Extract the 4 execution type branches from `PositionStateMachine` into pluggable modules. Build an Execution Types management page.

**Tasks:**
- [ ] 2a. Define `ExecutionType` interface: `evaluate(signal, bar_data, position_state, params) → ExecutionAction`
- [ ] 2b. Extract Bar Close (C) execution logic into `BarCloseExecution` class
- [ ] 2c. Extract Level (L) execution logic into `LevelExecution` class
- [ ] 2d. Extract Level-Close (LC) execution logic into `LevelCloseExecution` class
- [ ] 2e. Extract Close-Close (CC) execution logic into `CloseCloseExecution` class
- [ ] 2f. Refactor `PositionStateMachine` to call execution type modules instead of switch/if branches
- [ ] 2g. Create execution type registry (similar to pack_registry) — load, register, list
- [ ] 2h. Build Execution Types page in frontend — list, create, edit parameters, enable/disable
- [ ] 2i. Isolation testing: verify each execution type produces correct trades independently (synthetic signal → expected fill)
- [ ] 2j. Verify: existing strategies produce identical trades after refactor (no regression)
- [ ] 2k. Update Ralph engine to use the same execution type modules (parity with unified engine)

**Exit Criteria:** Execution types are pluggable modules. Adding a new execution type requires no engine changes. Each type is testable in isolation. Ralph engine and unified engine use the same execution modules.

---

### Milestone 3: Webhook Pipeline for Next.js
**Priority:** High — required for live trading
**Effort:** 3-5 days

Wire the webhook/alert system for the Next.js frontend. The Streamlit-era webhook system needs to be connected to the new API layer.

**Tasks:**
- [ ] 3a. Wire alert monitor enable/disable from Next.js frontend (currently Streamlit-only)
- [ ] 3b. Wire webhook template CRUD (create, edit, test) in Next.js settings/webhooks page
- [ ] 3c. Connect Ralph engine alert dispatch to webhook delivery system
- [ ] 3d. Add execution type to alert records and webhook payload templates
- [ ] 3e. Test end-to-end: strategy triggers → alert record → webhook fires → Discord/broker receives
- [ ] 3f. Account-level webhook templates (per-user defaults)

**Exit Criteria:** User can enable alerts on a strategy, configure webhook URLs, and receive real-time notifications when triggers fire. Webhook payload includes execution type, fill price, and position details.

---

### Milestone 4: Pack Builder Polish
**Priority:** Medium — reliability of pack creation
**Effort:** 3-5 days

Ensure packs created through the Pack Builder are consistently reliable and work seamlessly across all features.

**Tasks:**
- [ ] 4a. Fix remaining validation gaps (from testing — add checks as discovered)
- [ ] 4b. Update `pack_builder_context.md` with golden child reference (EMA PP V2 as template)
- [ ] 4c. Add Swing 1-2-3 as Pack Builder test case — validate CC triggers work end-to-end
- [ ] 4d. Pack status workflow: Verification → Private → Public (persist status, gate Strategy Builder visibility)
- [ ] 4e. Pack versioning: installed pack is immutable, new versions create new packs
- [ ] 4f. Delete protection: warn if strategies reference the pack
- [ ] 4g. Legacy pack cleanup: mark old TF Confluence packs as "Legacy (Default)"
- [ ] 4h. **Chart Preview tab** — Wire state-colored background + trigger markers (like Streamlit). Remove duplicate chart module. Show conditions as background color changes, triggers as markers on candles.
- [ ] 4i. **Signal Validation tab** — Wire real data: run indicator on sample data, count trigger fires, verify all states reached, per-trigger breakdown
- [ ] 4j. **Parity Simulator tab** — Nice-to-have. Backtest ↔ live engine parity comparison. May be complex; evaluate feasibility.
- [ ] 4k. Create several user packs through the full Pack Builder flow and verify each one end-to-end

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
