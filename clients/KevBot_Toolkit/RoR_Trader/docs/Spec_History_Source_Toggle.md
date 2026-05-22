# Spec — Chart & Trades History Source Toggle

**Status:** IMPLEMENTED 2026-05-21 (Strategy Detail → Chart & Trades tab).
Revised same day — "Backtest Model" label + trades-table data source.

## 1. Why

The Chart & Trades tab shows two trade tables side by side:

- **Left** — "Algo History": the live algo lane (`cache_%` rows in the
  `trades` table — what the live algo engine actually produced).
- **Right** — "Alert History": paired entry/exit alerts.

The Δ columns measure how closely the live alerts track the algo lane.
That answers *one* question — "is the live engine firing correctly vs
the decision-time data it saw?" (the accountability check).

It does **not** answer the question Kevin actually cares about most:
"are the live alerts firing close to the **backtest**?" The backtest
(`backtest_%` lane, `rest_hifi`) is the canonical target — the thing
live trades should ultimately converge toward. That comparison existed
only on the Divergence tab, not inline on Chart & Trades.

## 2. What

A **left-pane source toggle** on the Chart & Trades history module —
modelled on the Parity Simulator's left/right pane selectors. The right
pane is always Alert History; the left pane switches between:

| Mode | Left pane | Δ columns measure |
|---|---|---|
| **Algo Model** (default) | live algo lane (`cache_%`) | alert ↔ algo — live-engine accountability |
| **Backtest Model** | backtest-model lane (`backtest_%`) | alert ↔ backtest — fidelity-to-target |

Both the left table's Δ column and the right (Alert History) table's Δ
column re-pair against whichever lane is selected. The Alert History
header shows `Δ vs Algo Model` / `Δ vs Backtest Model` so the
comparison basis is never ambiguous.

Default is **Algo Model** — unchanged behaviour for anyone not touching
the toggle.

### Label note — "Backtest **Model**"

The button says "Backtest **Model**", not just "Backtest". Bare
"backtest" is ambiguous: it collides with the backtest-vs-forward-test
*time split*. "Backtest Model" makes it unambiguous — it refers to the
`backtest_model` data lane, the sibling of `algo_model` and
`live_model`. (Matches the Backtest/Algo Models admin page wording.)

### Data source — trades table, not the JSONB blob

The Backtest Model pane reads the **`trades` table** (`data_source LIKE
'backtest_%'`), NOT the strategy's `stored_trades` JSONB. The JSONB blob
lags: the recompute cron appends new backtest trades to the *table*,
not the blob. Example (sid 150, 2026-05-21): `stored_trades` JSONB held
3,374 rows ending 2026-04-29, while the trades table held 7,571
`backtest_rest_hifi` rows current to that day. Reading the table keeps
the pane current to today and consistent with the algo lane (which
already reads the table).

## 3. Implementation

All in `frontend/src/views/StrategyDetailPage.tsx`. No API or engine
change — both lanes were already loaded (`algoTrades`, `btTrades`).

- `historyLeftSource: 'algo' | 'backtest'` — `useState`, default `algo`.
- `backtestLaneTrades` — backtest-model lane loaded from the trades
  table via `useStrategyAlgoTrades(id, null, 'backtest_%')`. The
  `/algo-trades` endpoint gained a `data_source_filter` query param
  (`cache_%` default | `backtest_%`) so one endpoint serves both lanes.
- `compareTrades` — `useMemo` resolving to `algoTrades` or
  `backtestLaneTrades`.
- `computeMatches(compareSet, alerts, slipTol, timeframeMs)` — the
  nearest-neighbour entry/exit pairing, **extracted** from the old
  `algoMatches` useMemo into a source-agnostic helper. Returns
  `{ alertMatches, tradeMatches }`.
- `algoMatches` — still computed from `algoTrades` only. The Lab tab's
  Price Divergence panel indexes `algoMatches` against `algoTrades`, so
  it must NOT follow the toggle — kept as a separate fixed memo.
- `historyMatches` — `computeMatches(compareTrades, …)`; drives both
  history tables. In Algo mode it is identical to the old pair.
- Toggle UI — a 2-button segmented control in the left card header
  (Algo / Backtest), styled like the existing `labDataSource` toggle.

## 4. Out of scope

- The Divergence tab (already does three-way live↔algo↔backtest).
- The Lab tab (its own Algo Lens / Alert Lens panes).
- Right-pane source switching — the right pane is always Alert History;
  alerts are the fixed reference both lanes are judged against.

## 5. Possible follow-ups

- Persist the toggle choice in `chartPrefs` so it survives navigation.
- A third left-pane option for the forward-test lane (`fwdTrades`).
