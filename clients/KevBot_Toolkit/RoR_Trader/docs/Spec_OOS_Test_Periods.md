# Spec — Out-of-Sample Test Periods & Mass Builder OOS Gate

**Status:** DRAFT 2026-05-21 — awaiting Kevin review before implementation.
**Backup branch:** `dev-backup-2026-05-21-pre-oos` (`ff9644f`).

## 1. Why

Mass Builder today optimizes a strategy over one window and ranks by
KPI. Nothing distinguishes a real edge from a curve-fit to that
window's noise. The fix is the standard quant discipline: split history
into an **in-sample** window (optimize here) and an **out-of-sample**
window (validate here, never optimized on). A strategy that holds up
out-of-sample is trustworthy; one that collapses was overfit.

This is **not** a rolling walk-forward — one in-sample/OOS split per
strategy, fixed at build time.

## 2. The model — two orthogonal axes

A strategy's trade history is described by two independent axes.

### 2.1 Time axis — three bands

| Band | Span | Optimized on? |
|---|---|---|
| **In-Sample** | `in_sample_start` → `in_sample_end` | YES — the optimizer scored KPIs here |
| **OOS-Historical** | `in_sample_end` → `forward_test_start` | No — held-out historical data |
| **Post-Creation** | `forward_test_start` → now | No — real elapsed time since the strategy was saved |

### 2.2 Lane axis — the data source

| Lane | Source | Exists for |
|---|---|---|
| **Backtest model** | `trades` table, `data_source LIKE 'backtest_%'` | all three bands — the **consistent backbone** |
| **Alerts** | `alerts` table | only where alerts exist (≈ Post-Creation) |

**Decision (Kevin, 2026-05-21):** every computed time band — In-Sample,
OOS-Historical, Post-Creation — is computed on the **backtest model**
lane. The **algo model is NOT a band source**; it stays as the
live-accountability mirror behind the Chart & Trades history toggle and
the Divergence tab, and is out of scope for this feature.

### 2.3 The grid

"Live" is not a fourth band — it is the **Alerts lane ∩ Post-Creation
band**. "Forward" data and "Live" data cover the *same dates*; the gap
between the backtest-model lane and the alert lane over that span is the
execution-fidelity signal (a discrepancy implies a live-execution issue,
not a backtest issue). That comparison already exists — Divergence tab,
Chart & Trades history toggle — and is unchanged by this spec.

## 3. Terminology — keep "Backtest" / "Forward Test"

We do **NOT** rename anything user-facing or in code. The existing
vocabulary maps onto the new model:

- **"Backtest"** ≡ the In-Sample band.
- **"Forward Test"** ≡ OOS-Historical + Post-Creation.
- **OOS-Historical** is surfaced as a distinctly-colored *sub-band*
  inside the Forward Test region — visible, not obfuscated, but it rolls
  up into the Forward KPI bucket.

Rationale: renaming `backtest`/`forward_test` across the codebase would
make future debugging treacherous (a log line saying "based on the
backtest" could mean either the old or new sense). Stable names, one
moved boundary — see §5.

## 4. Data model

Two roles, currently both served by `forward_test_start`, must be
**separated**:

| Role | Field | Notes |
|---|---|---|
| Strategy creation anchor / alert-tracking anchor / divergence windowing | `forward_test_start` | **UNCHANGED, immutable.** Keeps every current role except the KPI-period divider. |
| In-Sample → Forward KPI divider | **`in_sample_end`** (NEW) | The end of the optimizer's window. |
| In-Sample window start | **`in_sample_start`** (NEW) | The optimizer's window start; for charting the In-Sample band. |

- New fields live in `strategy.config` (JSONB) — additive, no schema
  migration. (Heed `feedback_jsonb_partial_updates` — never round-trip a
  partial config dict through `_strategy_to_row`.)
- **Default when unset:** `in_sample_end` resolves to
  `forward_test_start`; `in_sample_start` resolves to the first
  backtest-lane trade's entry date. So every existing strategy and every
  strategy built the normal way (backtest end = today) has a
  **zero-width OOS-Historical band** and behaves **byte-identically** to
  today.
- **No `live_date` field.** "Live" is self-defining as "where alerts
  exist" — derive effective go-live from the first alert timestamp.
- `forward_test_start` stays immutable — see
  `feedback_forward_test_start_immutable`. Nothing here resets it.

### 4.1 Period resolver — the single source of truth

Add one helper, used everywhere a trade must be bucketed:

```
resolve_in_sample_end(strategy)  -> in_sample_end or forward_test_start
classify_trade_period(entry_ts, strategy) -> 'in_sample' | 'oos_hist' | 'post_creation'
```

All KPI splits, chart bands, and the Mass Builder gate go through these.
No ad-hoc date comparisons.

## 5. The boundary shift — critical audit

Today the Backtest|Forward KPI split happens at `forward_test_start`.
It must move to `resolve_in_sample_end(strategy)`. Because the resolver
defaults to `forward_test_start`, **this is behavior-neutral for every
strategy that does not have an explicit `in_sample_end`** — i.e. all
existing strategies and all normally-built ones. Only strategies built
through the Mass Builder OOS flow see the new split.

`forward_test_start` has **multiple roles** (≈100+ usages). Only the
**KPI-period-divider** role moves. Other roles (creation anchor, alert
tracking, divergence windowing) keep using `forward_test_start`.

**Audit checklist — inspect each, repoint ONLY divider usages:**

- `src/services.py` (18) — KPI computation core.
- `src/api/routers/strategies.py` (29) — strategy KPI endpoints.
- `src/api/services/forward_test_service.py` (6) — BT/FWD trade split.
- `src/api/services/divergence_service.py` (6) — *windowing role; likely
  unchanged.*
- `src/portfolios.py` (10) + `src/api/routers/portfolios.py` (5) —
  portfolio-level KPI rollups.
- `src/db.py` (4).
- `src/alerts.py` (2) — *alert-tracking role; unchanged.*
- `src/api/services/parity_service.py` (2), `pack_canary_service.py` (4).
- `src/app.py` (39) — legacy Streamlit; repoint only if still relied on.
- Frontend: `StrategyDetailPage.tsx`, `StrategiesPage.tsx`,
  `PortfolioDetailPage.tsx`, `useStrategies.ts`,
  `useStrategyMutations.ts`. (`versions/V2–V4.tsx` are snapshots — skip.)
- One-off `_*.py` audit scripts — ignore.

Each repointed site gets a comment: `# OOS: in_sample_end divider`.

## 6. Charts — banded equity curve

The equity curve currently splits at `forward_test_start`. Add an
**additive** split at `in_sample_end` — a new amber band inserted
*before* the existing forward region (the existing forward orange + the
green alert overlay are untouched, so pre-OOS charts are unchanged):

| Band | Color | Boundary |
|---|---|---|
| In-Sample (Backtest) | blue `#2196F3` | `in_sample_end` |
| **OOS-Historical** | **amber `#FFC107` (NEW)** | `forward_test_start` |
| Forward | orange `#FF9800` | (alert start) |
| Alert overlay | green `#4CAF50` | — |

`EquityCurve` gains an optional `oosBoundaryIndex` prop. When null
(every pre-OOS strategy — no `in_sample_end`), the amber band has zero
width and the chart renders exactly as before. The legend shows an
"Out-of-Sample" chip only when the band is present.

**Implemented in Phase 1** (`EquityCurve.tsx`, `StrategyDetailPage.tsx`).

Surfaces: Strategy Detail equity curve + price chart, forward-test view,
My Strategies card mini-charts, Portfolio Detail equity curves.

## 7. My Strategies — KPI filter

The existing filter modes are **kept**; only their bucket boundary
sharpens (via §5):

| Mode | Definition |
|---|---|
| **Overall** | backtest-model KPIs over **all** computed trades (In-Sample + OOS-Hist + Post-Creation). No alerts — duplicative. |
| **Backtest vs Forward** | backtest-model: In-Sample bucket vs (OOS-Hist + Post-Creation) bucket. Split at `in_sample_end`. |
| **Forward vs Alerts** | backtest-model Forward bucket vs alert lane. |
| **Backtest vs Alerts** | backtest-model In-Sample bucket vs alert lane. |

Card headline KPIs follow whichever mode is selected; default Overall.

## 8. Strategy Detail page

- Equity/price charts gain the amber OOS-Historical band (§6).
- KPI panels reuse the §7 modes.
- Optional (Phase 2): an **In-Sample vs Out-of-Sample robustness
  readout** — side-by-side key KPIs + a verdict chip (holding / degraded
  / overfit) driven by the §9.3 sigma test. This is the at-a-glance
  "did the edge survive?" panel.

## 9. Mass Builder — the OOS gate

### 9.1 Windows

- The user picks the **in-sample** date range (start + end) — this is
  today's single date range, re-labeled "In-Sample".
- The **OOS window auto-fills** to `in_sample_end → today` (editable but
  defaulted). If the in-sample end is today, the OOS window is empty and
  the gate is inert — today's behavior.
- The engine backtests each combo over the **full** range
  (`in_sample_start → today`); KPIs are then computed twice — once over
  the in-sample sub-range, once over the OOS sub-range. Cheap: the
  trades are already produced by Layer 1 (see M7 architecture —
  `project_m7_mass_builder_wip`). No extra engine passes.

### 9.2 The gate is two-sided and lives in Required Performance

Mass Builder keeps only a capped number of results, so an overfit combo
can rank high in-sample and squeeze a robust one out. Therefore the OOS
check is a **Required-Performance gate**, not just a sort column:

> A combo is **kept** only if it clears the **in-sample** minimums
> **AND** the **OOS** minimums.

The OOS Required-Performance section is **optional** — leave it blank
and Mass Builder behaves exactly as today (in-sample gating only).

### 9.3 Two gate types — build both

The OOS Required-Performance section supports two independent,
combinable gate types; the user fills in whichever they prefer:

1. **Raw thresholds** — plain minimums on OOS KPIs: OOS win-rate ≥ X,
   OOS profit-factor ≥ Y, OOS daily-R ≥ Z, OOS min trade count, etc.
   Simple, explicit.
2. **Sigma band** — the truer overfit detector. From the in-sample
   per-trade R stats (mean μ, std σ, n), project the expected OOS
   cumulative-R curve `k·μ` with band `±N·σ·√k` (reuses the Phase 37
   Performance-vs-Plan confidence-band machinery —
   `project_m8_portfolios_wip` / Phase 37 spec). The combo's actual OOS
   curve must stay within ±Nσ (N configurable, default 2). A combo whose
   OOS result falls below the lower band = degrading → fails the gate.

Both gates AND together when both are filled.

### 9.4 Results table

Existing in-sample KPI columns + new OOS columns: OOS WR, OOS PF, OOS
daily-R, OOS trade count, and a **robustness chip** (green = OOS within
1σ / amber = within 2σ / red = beyond). All sortable — "is it
*maintaining* performance" is read by sorting/scanning OOS columns.

Run summary reports `N combos gated out by OOS` so the user sees the
gate did work.

### 9.5 Save

Saving a strategy from Mass Builder stamps `in_sample_start` /
`in_sample_end` from the search's in-sample window onto the strategy.

### 9.6 Rollout — evolve in place

No `MassBuilderV2` page, no env-var toggle. The OOS feature is additive:
no OOS window / blank OOS Required-Performance = today's behavior. One
codebase. The backup branch is the safety net. Old saved Mass searches
remain valid (they simply have no OOS window).

## 10. Existing strategies — no migration

No backfill. With `in_sample_end` unset → resolver returns
`forward_test_start` → zero-width OOS band → 2-band chart, unchanged
KPIs. Existing strategies render and compute exactly as today. The OOS
band only ever appears on strategies built through the new flow.

## 11. Phasing

| Phase | Scope | Risk |
|---|---|---|
| **1** | Data model (`in_sample_start/end` + resolver §4.1) + 3-band chart coloring (§6). Purely additive — zero-width band for all existing strategies. | Low — safe to ship tonight. |
| **2** | Boundary audit + repoint (§5); My Strategies / Strategy Detail KPI modes (§7–8); robustness readout. | Medium — the audit is the risk; resolver default makes it behavior-neutral. |
| **3** | Mass Builder OOS gate (§9) — windows, two-sided gate, both gate types, results columns. | Medium — additive to a working page. |

## 12. Out of scope

- Rolling / moving-window walk-forward (single split only).
- Renaming `backtest` / `forward_test` in code or UI.
- The algo-model lane — unchanged; not a band source.
- A stored `live_date` — derived from first alert, not stored.
- `forward_test_start` semantics — immutable, only its KPI-divider role
  is delegated to the resolver.

## 13. Open decisions for review

1. **Phase 1 tonight?** Data model + chart banding is additive and safe
   behind the resolver default — OK to land tonight, with Phases 2–3 to
   follow?
2. **Robustness chip thresholds** — green ≤1σ / amber ≤2σ / red >2σ.
   Acceptable, or different cut points?
3. **OOS window editability in Mass Builder** — auto-fill
   `in_sample_end → today` and lock it, or auto-fill but allow the user
   to shorten it? (Spec assumes auto-fill, editable.)
