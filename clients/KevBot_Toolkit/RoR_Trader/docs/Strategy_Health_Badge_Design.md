# Strategy Health Badge — Design

## Problem

Strategies in My Strategies don't surface data-quality state. The clearest case: strategies saved from Mass Builder are based on rapid-test KPIs (short window, simplified stops, no Hi-Fi) and shouldn't be trusted in production until re-backtested. Today there's no visual cue — Kevin has to remember which strategies came from Mass Builder.

The same problem generalizes: a strategy may also have stale KPIs after editing, reference a deleted confluence group, sit in a portfolio with no webhook configured, etc. A single visible signal across all of these prevents "I forgot to fix that" foot-guns.

## Decision summary

A small badge icon on each strategy card. Color = severity of the worst issue. Hover = tooltip with issue counts. Click = drawer with full list + one-click fix actions.

Both staleness modes are in scope:
- **Edit-stamps-stale** — any backtest-affecting edit sets `kpis_stale_since`; only a re-backtest clears it.
- **Time-based stale** — KPIs older than a threshold (default 60 days) flagged regardless of edits, to catch market-regime drift.

## Icon mechanics

| Severity | Icon | Meaning |
|---|---|---|
| 🟢 Healthy | check / clean badge | No issues |
| 🟡 Minor | yellow info | Usable, awareness only (short window, no Hi-Fi yet) |
| 🟠 Action recommended | orange warning | Should fix before trusting (rapid test, stale KPIs, broken config) |
| 🔴 Broken | red error | Can't run as-is (missing entry trigger, deleted pack) |

Worst issue determines the color. Tooltip on hover shows the count by severity, e.g. *"1 orange, 2 yellow — click for details"*. Click opens a Strategy Health drawer.

## Issue catalog

### Data quality (mutually exclusive — one badge per strategy, picks highest observed)

| Severity | Condition |
|---|---|
| 🟢 | Hi-Fi backtest + ≥30 forward-test trades |
| 🟢 | Hi-Fi backtest, no forward data yet |
| 🟡 | Full backtest, never run with Hi-Fi |
| 🟡 | Full backtest but window <30 days |
| 🟠 | **Rapid test** (Mass Builder save, no full backtest yet) |
| 🟠 | Stale backtest — config edited since last KPI run |
| 🟡 | Time-stale — KPIs computed >60 days ago (configurable threshold) |

### Configuration (additive — any can fire)

| Severity | Condition |
|---|---|
| 🟠 | References disabled or deleted confluence group |
| 🟠 | References a user pack that no longer exists in `user_packs/` |
| 🔴 | Missing `entry_trigger_confluence_id` (legacy strategy, can't re-run) |
| 🟡 | Stop or target config uses an `exec_type` the engine doesn't honor |

### Operational (additive)

| Severity | Condition |
|---|---|
| 🟡 | In a portfolio that has no `webhook_group_id` |
| 🟡 | `monitored=true` but the worker engine isn't currently watching |
| 🟠 | Recent alert fired but webhook delivery failed |
| 🟠 | Phantom or missed trade detected (alert-history vs algo-history divergence) |

## Storage / detection

### New fields on `strategies`

| Field | Type | Default | Set when |
|---|---|---|---|
| `data_source` | `'rapid' \| 'full' \| 'hifi'` | `'full'` (best guess for legacy) | On save by whichever path produced KPIs: Mass Builder writes `'rapid'`, normal backtest writes `'full'`, Hi-Fi backtest writes `'hifi'` |
| `kpis_stale_since` | timestamptz \| null | null | Set when any backtest-affecting field is edited; cleared on KPI recompute |
| `kpis_computed_at` | timestamptz \| null | last_backtest_at if known | Set when KPIs are written. Drives the time-based stale check |

Everything else in the catalog is **derived on-the-fly from current state** — no schema needed:
- Orphaned configs: cross-reference `entry_trigger_confluence_id` against active confluence groups + pack registry.
- Operational: read worker `monitor_status`, recent alert delivery rows, alert/algo divergence flag.

### Pure-function health computer

```python
def compute_strategy_health(
    strategy: dict,
    portfolio: dict | None = None,
    engine_status: dict | None = None,
    recent_alerts: list[dict] | None = None,
) -> StrategyHealth:
    """Derive a StrategyHealth report from current state. No DB writes."""
```

Returns:
```python
@dataclass
class StrategyHealth:
    severity: Literal['healthy', 'minor', 'action', 'broken']
    issues: list[HealthIssue]  # ordered by severity desc

@dataclass
class HealthIssue:
    severity: Literal['minor', 'action', 'broken']
    code: str                  # e.g. 'rapid_test_data', 'orphan_pack'
    title: str                 # short — for tooltip
    detail: str                # longer — for drawer
    fix_action: str | None     # 'run_full_backtest' | 'configure_webhook' | …
    fix_action_label: str | None  # button label
```

### Backfill plan

One-time script: for every strategy in the DB, set `data_source='rapid'` if it has a `mass_search_id` or matches a heuristic (e.g. trade count or window length below threshold), else `'full'`. `kpis_computed_at` defaults to `updated_at` if no better timestamp is recorded.

## UI surfaces

| Surface | Treatment |
|---|---|
| **My Strategies card** | Badge in top-right corner. Hover tooltip. Click opens drawer scoped to this strategy. |
| **Strategy Detail page** | Banner at top with same drawer inline (always expanded if any issue). |
| **Portfolio card** | Aggregated count chip — "3 strategies have issues" + per-severity dots. Click to expand the affected strategies. |
| **Bulk actions** | When multi-selecting strategies, if any are 🟠/🔴 show "Re-backtest selected" / "Run Hi-Fi on selected" actions. |

## MVP scope (one session)

1. Migration: add the three new strategy columns + backfill script.
2. `compute_strategy_health` pure function (covers data quality + config; defer operational to phase 2 since it needs cross-table queries).
3. Set `data_source` on the four save paths: Mass Builder save, full backtest save, Hi-Fi backtest save, Forward Test recompute.
4. Set `kpis_stale_since` in the strategy edit path.
5. Backend: include `health` in strategy detail and list responses.
6. Frontend: badge component on the My Strategies card + tooltip + drawer (info-only, no fix buttons).

**Out of MVP:** operational issues (worker watching, webhook delivery), portfolio aggregation, fix-action buttons, bulk actions.

## Phase 2

- Operational issues — needs queries against `monitor_status`, recent alert delivery rows, divergence flag.
- One-click fix-action buttons in the drawer (Run Full Backtest, Run Hi-Fi, Configure Webhook for portfolio, etc.).
- Portfolio card aggregation chip + expand-affected-strategies.
- Bulk actions in My Strategies multi-select.

## Open questions to resolve at kickoff

1. **Time-stale threshold.** 60 days is a starting guess. Worth checking against typical Mass Builder windows + how often Kevin re-validates strategies in practice.
2. **Backfill heuristic for legacy strategies.** Is there a reliable signal that a strategy came from Mass Builder pre-`data_source` field? `mass_search_id` would be the cleanest if it exists; otherwise heuristic (trade count, window).
3. **Forward-test threshold for the green tier.** ≥30 trades is a placeholder — could be tighter (need statistical-significance call from Kevin) or looser.
4. **Edit detection scope.** Which strategy fields count as "backtest-affecting" for the `kpis_stale_since` stamp? Likely: entry/exit trigger, confluence list, stop/target config, time exit, secondary TFs, direction, session. Likely NOT: name, tags, description, portfolio assignment, monitored flag.
5. **Drawer copy.** Each issue needs a one-line tooltip + a longer drawer description. Worth a design pass before the first commit.

## Critical files (anticipated, not yet read)

- `src/db.py` — strategy CRUD + new column reads/writes
- `src/services.py` — central place for the `compute_strategy_health` pure function
- `src/mass_builder.py` — set `data_source='rapid'` on save (lines ~838, 907 per Phase 40 plan doc index)
- `src/api/services/backtest_service.py` — set `data_source='full'` or `'hifi'` post-run
- `src/api/services/forward_test_service.py` — `recompute_and_persist_stored_trades` updates `kpis_computed_at`
- `src/api/routers/strategies.py` — include `health` in detail/list response
- `src/app.py` — Streamlit edit handlers stamp `kpis_stale_since`
- `frontend/components/strategy_card/*` — badge component (Next.js)
- `src/migrations/phaseXX_strategy_health.sql` — schema migration (NEW)

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Backfill miscategorizes legacy strategies | One-off correction script; user can manually re-stamp via "Run Full Backtest" |
| Edit detection too aggressive — flags every minor change | Whitelist of backtest-affecting fields (open question #4) |
| Drawer becomes a wall of text | Keep MVP at info-only; design pass on copy before phase 2 fix-action surface |
| Health computation is expensive on list endpoints | Pure function; takes already-loaded strategy dict; no extra DB queries in MVP |
| Time-stale fires on backtests Kevin manually validated as still-good | Add a "mark as validated" action in phase 2 that resets `kpis_computed_at` without rerunning |

## Effort estimate

- MVP: ~4–6 hours (migration + backfill + helper function + 4 save-path stamps + edit hook + 2 API responses + 1 React component + drawer)
- Phase 2: another ~4–6 hours

## Why this matters

Strategy quality is currently tribal knowledge — Kevin remembers which strategies came from Mass Builder. As the strategy count grows (Mass Builder produces dozens per run), tribal knowledge stops scaling. A single visible signal lets the user trust the My Strategies list as a source of truth without mental bookkeeping.
