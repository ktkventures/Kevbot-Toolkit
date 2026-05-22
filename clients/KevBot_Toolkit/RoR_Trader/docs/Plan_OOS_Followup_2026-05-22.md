# EOD Plan — OOS feature shipped, follow-up for 2026-05-22

**Context:** The Out-of-Sample (OOS) test-period feature shipped end to
end overnight (2026-05-21 → 05-22). Spec: `Spec_OOS_Test_Periods.md`.
It works, but browser QA surfaced two issues that need iteration before
the feature is trusted. Kevin wants to look at this before the weekend.

## What shipped (all on `dev`, deployed)

- **OOS Phase 1** (`bbf4776`) — `strategy_periods.py` period resolver;
  amber OOS band on the Strategy Detail equity curve.
- **OOS Phase 3** (`669b9f5`) — Mass Builder OOS gate: in-sample/OOS
  split, two-sided Required-Performance gate (raw thresholds + sigma
  band), OOS KPI columns, robustness chip.
- **OOS Phase 2** (`6d2e0a7`) — KPI Backtest|Forward divider repointed
  to `resolve_in_sample_end` (behaviour-neutral pre-OOS).
- OOS sort options (`f402597`); card banding (`71f2c2e`, `15196ff`).
- Bug fixes from QA: `ddbf27d` (end_date override broke data load),
  `921bad5` (split_trades_at_boundary entry_time KeyError), `50d4ebf`
  (saveResult dropped in_sample_end).
- General Packs variation parameters made editable (`41f56d1`).

## OPEN ISSUE A — Mass Builder preview KPIs diverge from recompute

**Symptom:** A Mass Builder result showed profit factor **2.77** with a
great equity curve. After "Update All Data" on the saved strategy
(sid 244), the recomputed profit factor was **1.24** — a very different
curve.

**Root cause (known M7 limitation, not an OOS bug):** Mass Builder runs
one base backtest, then the confluence search is a *post-hoc boolean
mask* on that base run's trades. The full-engine recompute runs the
strategy with the confluence gate active from the start — different
trades get taken (position-state blocking differs). The MassBuilderPage
already carries a "preview KPIs" warning banner about exactly this. The
PF 2.77 → 1.24 here shows the gap can be large.

**Why it matters now:** the **OOS gate also runs on these preview
KPIs** — so a combo can clear the OOS gate on optimistic preview numbers
and then collapse on recompute. The OOS validation is only as honest as
the preview.

**Decide tomorrow:** (a) surface the preview-vs-real gap explicitly in
results, (b) confluence Hi-Fi (full-engine per combo — the deferred
M8.6 item), or (c) recompute the top-N before applying the OOS gate.

## OPEN ISSUE B — OOS band missing after "Update All Data"

**Symptom:** sid 244's Strategy Detail equity curve shows no in-sample/
OOS separation; the My Strategies card shows no band either.

**Root cause (confirmed via DB):** sid 244 **does** have
`config.in_sample_end = 2026-04-22` (the `saveResult` fix worked). But
its **`stored_trades` JSONB is empty (0)** — "Update All Data" populated
the **`trades` table** (339 backtest + 339 cache rows) and left the
JSONB blob empty. The Phase 1 chart boundary logic (`equityBoundaryIndex`,
`equityOosBoundaryIndex`, `equityPoints`) all read `stored_trades`
JSONB — empty blob → no boundary indices → no bands. This is the
Phase-40 trades-table-migration tension: trades moved to the table,
but a lot of the chart code still reads the JSONB.

**Fix direction tomorrow:** the chart boundary indices should derive
from the `trades` table / `equity_curve_data` (which has timestamps),
OR the recompute path should also repopulate the `stored_trades` JSONB.
Pick one — this affects every Mass-Builder-recomputed strategy, not just
OOS ones.

## Tomorrow — market-open monitoring (Kevin's priority)

- A couple hours after market open, check **backtest ↔ live alignment**
  — do live alerts fire close (timing) to where the backtest model says
  they should? This is the timing-fidelity read, separate from KPIs.
- Likely can't get a full clean trading day; monitor the first few
  hours and decide from there.

## Backup branches

`dev-backup-2026-05-21-pre-oos` (ff9644f), `-oos-spec` (d314fa4),
`-pre-oos-p3` (bbf4776), `-pre-oos-p2` (669b9f5),
**`dev-backup-2026-05-22` (41f56d1)** — latest, pre-follow-up.

## Open tasks

#43 serialize mass searches (concurrent runs contend for CPU — observed),
#39 persist recompute/mass jobs across deploys, #38 UI pack-delete
durability.
