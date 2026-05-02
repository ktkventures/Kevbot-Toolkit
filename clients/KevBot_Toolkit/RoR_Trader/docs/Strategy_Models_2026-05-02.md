# Strategy Models — placeholder phase

**Created:** 2026-05-02
**Status:** Placeholder — selection recorded on strategies; engine dispatch not yet wired.

## Why models exist

Throughout development we've repeatedly hit the algo-vs-live data divergence problem:
backtest reads Polygon REST (settled, includes late-print corrections, ≈ TradingView);
the live engine reads Polygon WebSocket (real-time, uncorrected at decision moment).
For knife-edge triggers on sub-minute timeframes these diverge enough to flip
indicator-cross signals, producing phantom and missed trades.

The "model" abstraction makes this divergence an **explicit, declared property of each
strategy** rather than a hidden assumption. A strategy authored under `backtest_model='rest_only'`
keeps that semantic forever, even if defaults shift later. Strategies built today don't
get retroactively broken by tomorrow's architecture changes.

## Two model fields

Each strategy carries two model fields in `config`:

### `backtest_model` — what data the algo-history view uses

**Safe to change.** Affects analytics only — does not change live execution.

| Value | Available | Description |
|---|---|---|
| `rest_only` | ✅ default | Polygon REST settled bars only (current behavior) |
| `rest_hifi` | ✅ | REST + Hi-Fi Pass 2 (1-second refinement of timestamps/prices) |
| `rest_with_cache_overlay` | ⏳ coming soon | REST for pre-cache periods, WS cache for post-cache (closest to "what live actually saw") |
| `cache_only` | ⏳ coming soon | Only periods since cache started (forward-only analysis) |
| `cache_first` | ⏳ coming soon | cache.first_close — values at decision moment, before any rebroadcast corrections |

### `live_model` — how the live engine handles WS rebroadcasts

**Change with care.** Affects alert firing in production.

| Value | Available | Description |
|---|---|---|
| `ws_with_corrections` | ✅ default | Apply Polygon WS rebroadcast corrections within 15-min FINRA window via recompute-from-history (Option B) |
| `ws_first_lock` | ⏳ coming soon | Lock bars at first WS write; ignore rebroadcasts (Option C — awaiting Monday TV stability test) |

## Where you set models

- **Strategy Detail page → Configuration tab → Models card**: dropdowns to change either
  field for an existing strategy. Saves immediately on change.
- **Strategy Detail page header**: small read-only badges (`BT: rest_only`, `Live: ws_with_corrections`)
  showing the active models without having to navigate.

## What's NOT yet wired (today)

- `recompute_and_persist_stored_trades` ignores `backtest_model` — always uses REST.
- The live engine ignores `live_model` — always uses Option B (post-2026-05-02 fix).
- Different strategies with different model values currently produce identical results.

The fields are stored and displayed today so that:
1. New strategies created today carry their intended model in config.
2. When the cache read path (M8.7d) lands next week, the wiring just reads the stored
   value — no per-strategy migration needed.
3. The user has time to think in model terms while the foundation is being built.

## Cross-analysis vision

Once both fields drive behavior, the matrix of (backtest_model × live_model) defines
8+ combinations per strategy. Comparing forward-test KPIs across combinations will
empirically answer:
- Does WS-cache backtest predict forward results better than REST backtest, on average?
- Are sub-minute strategies more sensitive to the choice than 5Min+ strategies?
- Does Option B (ws_with_corrections) actually outperform Option C (ws_first_lock) in
  forward tests?

These are questions we currently cannot answer because all strategies share the same
hidden defaults. With the model abstraction, the question becomes a per-strategy
configurable variable that we can systematically vary and measure.

## Why this is structurally different from a feature flag

Feature flags are global — flipping one changes behavior for all strategies at once.
The model abstraction is per-strategy — strategies can disagree about what data they
treat as authoritative. This matters because:
- Strategy A (5Min trend regime) doesn't care about WS-vs-REST drift — runs `rest_only`.
- Strategy B (10Sec EMA cross) is highly sensitive — runs `cache_first` and accepts
  the limited backtest window as the price of accuracy.
- Both coexist in the same My Strategies list with different declared semantics.

A feature flag would force them to share semantics. The model abstraction lets each
strategy carry its own.

## Roadmap from placeholder to active

1. **2026-05-02 (today)**: schema fields exist; UI surfaces them; engine ignores them.
2. **Next session**: M8.7d ships cache read path. `backtest_model='cache_*'` starts
   working. `recompute_and_persist_stored_trades` reads `backtest_model` and dispatches.
3. **TBD**: live engine starts respecting `live_model`. Likely after Monday's TV
   stability test informs the lock-vs-corrections choice.
4. **Eventually**: more model values added (e.g., dual-source overlay; per-tick Hi-Fi).
   Existing strategies stay valid; new options just appear in the dropdown.

## Critical files

| Concern | File |
|---|---|
| Model constants | `src/strategy_models.py` |
| Models endpoint | `src/api/routers/strategies.py` (search `list_strategy_models`) |
| Default fill on get | `src/api/routers/strategies.py` (search `MODELS default fill`) |
| Frontend hook | `frontend/src/hooks/queries/useStrategies.ts` (`useStrategyModels`) |
| Models card UI | `frontend/src/views/StrategyDetailPage.tsx` (`ModelsCard`) |
| Header badges | `frontend/src/views/StrategyDetailPage.tsx` (search `M8.7 (2026-05-02) Models placeholder badges`) |
