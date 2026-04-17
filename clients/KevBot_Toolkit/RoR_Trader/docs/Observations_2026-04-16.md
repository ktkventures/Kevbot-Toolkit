# Session Notes — 2026-04-16

End-of-session observations. Pick these up on 2026-04-17.

---

## Ship log (this session)

| Commit | Area |
|---|---|
| `7ac9189` | Indicator param resolution: strict contract + admin-client fallback for worker thread |
| `fea6e4a` | Confluence Analysis chart: filter non-plottable columns + memoize panes |
| `86b7c9d` | SyncedChartPane: honor user display timezone on axis + crosshair |
| `b83629b` | Algo history architecture: `stored_trades` = backtest truth, only |
| `a0e9586` | Refresh endpoint: `user` is a dict, not an object |
| `0344617` | L-type fills clamped to bar range (no more off-bar `+` markers) |
| `0e768be` | Ghost line for v2 L-type triggers (`utbot_stop_prev`, `ema_N_prev`) |

Backup branch: `dev-backup-pre-algo-history` covers the state before the whole algo-history refactor.

---

## Open observation — Suspected 1-bar lag on L-type entries (strategy 114)

**Symptom (from image copy 95):**
User saw an entry around 17:42:20 where the `+` marker sat on the dashed ghost line (`utbot_stop_prev`), but the candle of the entry bar itself was *below* that line. Meanwhile price had crossed above the (solid) `utbot_stop` a candle or two earlier. Feels like the fill is one bar too late.

**Pointer — likely NOT a bug, but worth confirming:**
For v2 triggers (`utbot_v2_buy_ib`), the crossing level is the PREVIOUS bar's `utbot_stop`. That level is static for the duration of the current bar. The trigger fires when the current bar's high crosses that static level — so by definition the fill is on the bar AFTER the level was set.

For v1 triggers (`utbot_buy_ib`), level is the current bar's `utbot_stop` — self-adjusts intra-bar, so crosses and fires land on the same bar. User has zero v1 strategies right now (all 11 UTBot strategies are v2).

**What to try tomorrow (before chasing it as a bug):**
1. Create a v1 utbot strategy (`utbot_buy_ib` entry, not `utbot_v2_buy_ib`). Compare marker timing on the same SPY 10-sec bars.
2. If v1 markers land on the cross bar and v2 markers land one bar later, it's design-correct — just surprising. Document the visual distinction.
3. If v1 ALSO looks shifted, we have a real timestamp bug. Start by tracing `entry_time` assignment in `PositionStateMachine.check_entry_intrabar` + `check_entry` for L-fills from `l_type_fills`.

**What it's NOT:**
The `shiftIfCType` helper in `buildStrategyChartPanes.ts` explicitly skips L and LC execs — no shift is applied. We confirmed this earlier in the session.

---

## Hi-Fi backtest status

Partially implemented. Current state:

- `_hifi_resolve_trades` in `api/services/backtest_service.py:576` — **active** Pass 2 that loads 1-second bars from Polygon for each trade's entry/exit window and replaces the coarser-bar fill with the 1-sec resolution.
- Triggered ONLY by `req.hifi_mode == True` on the `/api/backtest` endpoint (Strategy Builder path).
- **NOT wired into the stored_trades recompute path** (`forward_test_service.recompute_and_persist_stored_trades` → `services.get_strategy_trades`). So Chart & Trades in Strategy Detail + bar-close auto-refresh are still running on coarse bars.

**Why it matters for the "shifted" observation:**
Hi-Fi at 1-sec resolution would let the engine see the exact second price crossed the level. In extended-hours illiquid SPY, the coarse 10-sec bar averages multiple crossings; 1-sec resolution shows them. If the v2 "1-bar late" observation persists under Hi-Fi, it's design. If Hi-Fi resolves it, it was a granularity artifact.

**Wiring Hi-Fi into the stored_trades path is a one-line flip** inside `forward_test_service.recompute_and_persist_stored_trades` (pass `hifi_mode=True` into the backtest call). Costs more Polygon round-trips per recompute. Worth a deliberate decision before flipping.

---

## Degenerate bar visual issue (not a bug, but worth knowing)

10-sec SPY bars in extended hours are frequently O=H=L=C (single-print bars). The `+` marker sits on this "bar" correctly (the fill IS at that single price), but visually the marker appears disconnected from neighboring full-body candles. This is a market-data artifact, not an engine bug. 1-minute charts in RTH don't have this.

Mitigation options for later:
- Render single-print bars as wider horizontal tick marks so the marker lands inside visible geometry.
- Default the chart to a denser timeframe (1-min) and only offer 10-sec as an opt-in when tick data warrants.
- Hi-Fi (see above) sidesteps it for backtest fills.

---

## Roadmap nudges

- **Phase 31F (Hi-Fi stored_trades wiring)** — bump priority. The Execution_and_Fidelity_Playbook.md + Implementation_Spec_Phase_31F.md already spec most of it.
- **v1 vs v2 trigger education** — add a note to `Pack_Development_Workflow.md` or a new doc explaining the timing semantic difference. Users reading their own charts get confused without it.
- **Live chart auto-refetch of stored_trades** — the bar-close hook updates the DB but the frontend `useStrategy` query has no `refetchInterval`. User has to hard-refresh to see new algo markers.
- **Confluence Analysis tab ghost lines** — that chart has its own inline pane builder (not `buildStrategyChartPanes`). If user wants the same dashed-prev-line treatment there, it's a ~15 min port.

---

## Things user might try tomorrow

1. Create an L-type v1 strategy (utbot_buy_ib) for comparison.
2. Create a C-type 10-sec strategy for comparison (shift-to-next-open semantic on sub-minute bars).
3. Toggle Hi-Fi on a backtest run in Strategy Builder and see if the marker timing tightens.
