# Phase 38 Execution Plan — Source of Truth

**Status:** ACTIVE
**Date:** 2026-03-25
**Branch:** `dev`
**Backup:** `dev-backup-pre-38-impl`

---

## Approach

Copy each locked V5/V6/V7/V1 design file exactly into `views/`, then surgically replace mock data:

| Data State | Display | Meaning |
|-----------|---------|---------|
| Real API data | Actual value | Wired and working |
| `--` | Double dash | Wired to API but data not available yet |
| `{{field_name}}` | Curly brackets | Not wired yet — needs backend or feature work |

**Rule:** No fake numbers that look like real data. Ever.

### Lesson Learned (Batch 1 QA)

V5 files have TWO layers of mock data:
1. **Top-level constants** (e.g., `const strategy = {...}`, `const mockTrades = [...]`) — easy to find and replace
2. **Inline mock data in JSX** (e.g., hardcoded prices `142.35`, dates `2026-03-18 09:42:15`, table rows with fake trades) — hidden throughout tab content, much harder to find

The copy-and-wire procedure MUST include a `grep` audit for inline mock data AFTER replacing top-level constants. Patterns to search: specific prices (`14[0-9]\.`), specific dates (`2026-03-1[0-9]`), `mock` references, `Math.random`.

### Known Issues per Tab (Strategy Detail V5)

| Tab | Mock Data Status | What Needs Wiring |
|-----|-----------------|-------------------|
| Equity & KPIs | Top-level replaced, extended KPIs show 0 | Wire `useStrategyKPIs()` secondary_kpis to extendedKPIs |
| Chart & Trades | Trade tables wired but may show 0 FWD trades | Wire `useStrategyForwardTest()` properly; bar_count_exit = valid exit |
| Confluence Analysis | Entire tab is inline mock data (stateTimeline, triggerEvents) | Needs new API endpoint for per-group state/trigger history |
| Configuration | Mostly wired | bar_count_exit should display as exit trigger |
| Alerts | mockAlerts array + inline trade-to-alert mapping table | Wire `useStrategyAlerts()` to replace mockAlerts |
| Alert Analysis | mockAlertAnalysis with phantom trades, timing deltas, slippage | Needs new API endpoint for discrepancy/timing analysis |

---

## Phase A: Frontend Faithful Copy

### Batch 1: Core Flow
| Task | Source Design | View File | Status |
|------|-------------|-----------|--------|
| A1.1 | `strategies/versions/V5.tsx` (885 lines) | `views/StrategiesPage.tsx` | Not Started |
| A1.2 | `strategies/[id]/versions/V5.tsx` (2,177 lines) | `views/StrategyDetailPage.tsx` | Not Started |
| | | **Kevin QA checkpoint** | |

### Batch 2: Dashboard + Alerts
| Task | Source Design | View File | Status |
|------|-------------|-----------|--------|
| A2.1 | `dashboard/versions/V7.tsx` (2,492 lines) | `views/DashboardPage.tsx` | Not Started |
| A2.2 | `alerts/versions/V5.tsx` (450 lines) | `views/AlertsPage.tsx` | Not Started |
| A2.3 | `alerts/webhook-templates/versions/V5.tsx` (323 lines) | `views/WebhookTemplatesPage.tsx` | Not Started |
| A2.4 | `alerts/webhook-templates/[id]/versions/V5.tsx` (518 lines) | `views/WebhookTemplateDetailPage.tsx` | Not Started |
| | | **Kevin QA checkpoint** | |

### Batch 3: Confluence Packs
| Task | Source Design | View File | Status |
|------|-------------|-----------|--------|
| A3.1 | `tf-confluence/versions/V5.tsx` (1,855 lines) | `views/TfConfluencePage.tsx` | Not Started |
| A3.2 | `general/versions/V5.tsx` (813 lines) | `views/GeneralPacksPage.tsx` | Not Started |
| A3.3 | `user-packs/versions/V5.tsx` (1,743 lines) | `views/UserPacksPage.tsx` | Not Started |
| A3.4 | `pack-builder/versions/V7.tsx` (1,047 lines) | `views/PackBuilderPage.tsx` **NEW** | Not Started |
| A3.5 | `stop-loss/versions/V1.tsx` (541 lines) | `views/StopLossPage.tsx` | Not Started |
| A3.6 | `take-profit/versions/V1.tsx` (515 lines) | `views/TakeProfitPage.tsx` | Not Started |
| A3.7 | `timeframes/versions/V5.tsx` (192 lines) | `views/TimeframesPage.tsx` | Not Started |
| | | **Kevin QA checkpoint** | |

### Batch 4: Portfolios
| Task | Source Design | View File | Status |
|------|-------------|-----------|--------|
| A4.1 | `portfolios/versions/V5.tsx` (688 lines) | `views/PortfoliosPage.tsx` | Not Started |
| A4.2 | `portfolios/[id]/versions/V5.tsx` (1,820 lines) | `views/PortfolioDetailPage.tsx` | Not Started |
| A4.3 | `portfolios/new/versions/V5.tsx` (1,158 lines) | `views/PortfolioNewPage.tsx` **NEW** | Not Started |
| A4.4 | `portfolio-requirements/versions/V5.tsx` (685 lines) | `views/RequirementsPage.tsx` | Not Started |
| | | **Kevin QA checkpoint** | |

### Batch 5: Mass Builder + Settings
| Task | Source Design | View File | Status |
|------|-------------|-----------|--------|
| A5.1 | `mass-builder/versions/V6.tsx` (1,386 lines) | `views/MassBuilderPage.tsx` | Not Started |
| A5.2 | `mass-results/versions/V5.tsx` (241 lines) | `views/MassResultsPage.tsx` | Not Started |
| A5.3 | `settings/display/versions/V5.tsx` (1,353 lines) | `views/SettingsDisplayPage.tsx` | Not Started |
| A5.4 | `settings/connections/versions/V5.tsx` (426 lines) | `views/SettingsConnectionsPage.tsx` | Not Started |
| A5.5 | `settings/account/versions/V4.tsx` (276 lines) | `views/SettingsAccountPage.tsx` **NEW** | Not Started |
| A5.6 | `settings/profile/versions/V4.tsx` (215 lines) | `views/SettingsProfilePage.tsx` **NEW** | Not Started |
| | | **Kevin QA checkpoint** | |

### Strategy Builder
| Task | Source Design | View File | Status |
|------|-------------|-----------|--------|
| A0.1 | `strategy-builder/versions/V5.tsx` (2,494 lines) | `views/StrategyBuilderPage.tsx` | **DONE** (V5 copy with API hooks) |

---

## Phase B: Backend Alignment

### B1: Stop Loss / Take Profit Separation
| Task | Description | Status |
|------|------------|--------|
| B1.1 | `GET /api/packs/stop-loss` — filtered RM packs | Not Started |
| B1.2 | `GET /api/packs/stop-loss/templates` — stop-only templates | Not Started |
| B1.3 | `PUT /api/packs/stop-loss` — save via RM pipeline | Not Started |
| B1.4 | Repeat for take-profit | Not Started |
| B1.5 | `useStopLossPacks()` + `useTakeProfitPacks()` hooks | Not Started |

### B2: Execution Type Parameters
| Task | Description | Status |
|------|------------|--------|
| B2.1 | Extend TEMPLATES with exec variant schemas | Not Started |
| B2.2 | Update templates API serialization | Not Started |
| B2.3 | Update `ConfluenceTemplateDTO` TypeScript type | Not Started |

### B3: Fidelity Type [PB]/[CB]
| Task | Description | Status |
|------|------------|--------|
| B3.1 | Add fidelity_type to confluence condition records | Not Started |
| B3.2 | Default existing conditions to [PB] | Not Started |

### B4: Dashboard Endpoints
| Task | Description | Status |
|------|------------|--------|
| B4.1 | `GET /api/dashboard/equity-curve` | Not Started |
| B4.2 | `GET /api/dashboard/daily-pnl` | Not Started |
| B4.3 | `GET /api/dashboard/positions` | Not Started |
| B4.4 | `GET /api/dashboard/health` | Not Started |
| B4.5 | `GET /api/dashboard/activity` | Not Started |

### B5: Missing Strategy Fields
| Task | Description | Status |
|------|------------|--------|
| B5.1 | Forward test + alert KPIs on strategy endpoint | Not Started |
| B5.2 | Sigma deviation computation | Not Started |
| B5.3 | Strategy status derivation | Not Started |
| B5.4 | Equity curve data in strategy list | Not Started |

---

## Phase C: Feature Wiring

Replace `{{field_name}}` with real computed data.

### C1: Strategy Analytics
Forward KPIs, sigma badges, status, analysis tab computations, Markov, rolling metrics

### C2: Dashboard Widgets
Equity curve, daily P&L, P&L calendar, positions, health, regime, monthly goal

### C3: Portfolio Analytics
Equity curve, Monte Carlo, correlation, buying power, anomalies

### C4: Pack Configuration
Parameter editing, exec type tabs, CRUD operations

---

## Phase D: Future Pages + QA + Polish

### D1: Secondary Pages (admin, creator, marketplace, pricing)
All V4 copies with `{{field_name}}` throughout — no backend exists yet.

### D2: Visual QA
Kevin toggles PageSwitch on every page, logs discrepancies, I fix them.

### D3: Polish
Responsive, loading skeletons, error states, accessibility.

---

## Key Architectural Decisions

1. **Stop/Target:** Adapter endpoints over existing RM module — no DB migration
2. **Exec types:** Schema in TEMPLATES registry, stored in parameters JSONB
3. **Fidelity:** Derived from cross-TF shift logic — display only
4. **Dashboard:** New aggregation endpoints reusing existing modules
5. **views/ = live code, versions/ = frozen design reference**
