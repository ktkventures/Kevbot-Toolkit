# Phase H — Trade-Channel Shadow Bar Builder (2026-05-15)

## Context

Phase G's diagnostic phase ran today. Key finding: **`live_bars` cache decision-time view (`first_*`) hits ~80-84% close-match against Polygon REST**, and the post-rebroadcast view (`latest_*`) hits ~96%. The decision-time view IS what the engine uses to fire alerts, so 80% match means the engine's view of the market differs from settled reality on ~20% of bars at decision time.

Cause: Polygon's AM (1Min) and A (1Sec) WS channels emit pre-aggregated bars that get rebroadcast 20-100 seconds later with corrections (late-arriving trades). The engine fires alerts on the FIRST emission. We can't wait for rebroadcasts (too late for trading). The architectural fix is to **build bars ourselves from individual trades**, applying canonical filtering, with a small wait window (200-500ms) to capture in-flight trades — same approach professional quant firms use.

Polygon's T.{ticker} WebSocket channel streams individual trades live (included in Stocks Advanced plan, no extra cost). The `flat_file_ingestion.py` module already implements the right bar-building logic for batch — we adapt it for streaming.

**Intended outcome (Phase H.1)**: subscribe to T.* channels in shadow mode alongside existing AM/A. Build trade-based bars at three wait-time variants (0ms / 200ms / 500ms). Write to a new `live_bars_trades` table. Don't touch the production alert path. After 1-2 days of shadow data, evaluate via Bars Comparison and decide whether to promote one variant to be the alert source (Phase H.3).

**Risk profile**: zero impact on production alerts during shadow phase. Only adds reads from a new WS channel + writes to a new table.

## Strategy

- **H.1 tonight**: ship the shadow builder. New table, new WS subscription, three wait-time variants writing in parallel. Feature-flagged (`POLYGON_TRADE_SHADOW_ENABLED`, default ON for SPY+TSLA only — same scope as flat-file ingestion).
- **H.2 tomorrow**: extend Bars Comparison UI with 3 new sources for the wait-time variants. Backend endpoint reads `live_bars_trades`.
- **H.3 later (gated)**: if shadow data shows ≥95% close-match for a wait-time variant, promote it to the production alert path. Replaces `first_*` writes with trade-built bars.

## Standing Practices

- Backup branch: `dev-backup-pre-phaseH-2026-05-15` (already created)
- Commit + push at sub-phase boundary
- Feature flag `POLYGON_TRADE_SHADOW_ENABLED`, default `true`. Disable on Railway to roll back without redeploy.
- Symbol scope (initial): SPY + TSLA only (env-configurable via `POLYGON_TRADE_SYMBOLS`)
- Worker resource impact: ~150-300 trades/sec at peak across both symbols. Trivially handled in asyncio; no expected backpressure.

## Phase H.1 — Shadow trade-channel bar builder

### Files (new)

- **`src/migrations/live_bars_trades.sql`** — new table.
  - Schema mirrors `live_bars` except: PK includes `source` to allow multiple variants per bar.
    ```sql
    CREATE TABLE IF NOT EXISTS live_bars_trades (
      symbol             TEXT             NOT NULL,
      timeframe_seconds  INT              NOT NULL,
      bar_start          TIMESTAMPTZ      NOT NULL,
      source             TEXT             NOT NULL,   -- 't_wait0' | 't_wait200' | 't_wait500'
      open               DOUBLE PRECISION NOT NULL,
      high               DOUBLE PRECISION NOT NULL,
      low                DOUBLE PRECISION NOT NULL,
      close              DOUBLE PRECISION NOT NULL,
      volume             BIGINT           NOT NULL,
      trade_count        INT              NOT NULL DEFAULT 0,
      filtered_trades    INT              NOT NULL DEFAULT 0,
      written_at         TIMESTAMPTZ      NOT NULL DEFAULT now(),
      PRIMARY KEY (symbol, timeframe_seconds, bar_start, source)
    );
    CREATE INDEX IF NOT EXISTS live_bars_trades_lookup_idx
      ON live_bars_trades (symbol, timeframe_seconds, bar_start DESC);
    ALTER TABLE live_bars_trades DISABLE ROW LEVEL SECURITY;
    ```

- **`src/trade_bar_builder.py`** — streaming bar builder from individual trades. Adapts `_Bar` + filter logic from `flat_file_ingestion.py` and `polygon_conditions.py` for real-time use. One class `TradeBarBuilder` instantiated per `(symbol, tf_seconds, wait_ms)`.
  - Public API: `accept_trade(ev_dict)` — called once per Polygon T event for the symbol/tf this instance handles.
  - Calls `polygon_conditions._classify_eligibility(ev['c'])` for each trade (already exists, shared with flat-file ingestion).
  - Buckets by `floor(ev['t'] / 1e9 / tf_seconds) * tf_seconds` (sip_timestamp ns → epoch sec → bucket start).
  - Closes a bar when EITHER (a) a trade arrives in the next bucket, OR (b) wall-clock time exceeds `bucket_end + wait_ms` (timer-driven close for low-activity periods).
  - On close: write to `live_bars_trades` via Supabase admin client. Source label = `f't_wait{wait_ms}'`.

### Files (edited)

- **`src/ralph_engine.py`**
  - New feature flag helper `_polygon_trade_shadow_enabled()` (default ON, env `POLYGON_TRADE_SHADOW_ENABLED`).
  - In `_run_polygon_ws` (around line 3098): when the flag is on, add `T.{symbol}` channels to the subscription params for the symbols in `POLYGON_TRADE_SYMBOLS` (default `SPY,TSLA`).
  - In the message router (around line 3150 where `ev_type not in ('AM', 'XA', 'A', 'XAS')` filter is): handle `ev_type == 'T'` by dispatching to the per-symbol TradeBarBuilder instances. Each symbol has builders for each (tf, wait_ms) combo we want to shadow.
  - Initialization of TradeBarBuilder instances happens on first T event per symbol (lazy create), or at engine boot for the configured symbols.

### Configuration (env vars)

- `POLYGON_TRADE_SHADOW_ENABLED` — default `'true'`. Set to `'false'` on Railway Worker to disable.
- `POLYGON_TRADE_SYMBOLS` — default `'SPY,TSLA'`. Comma-separated.
- `POLYGON_TRADE_TFS_SECONDS` — default `'10,60'`. Which TFs to build. Sub-minute is the primary target; 1Min for cross-check.
- `POLYGON_TRADE_WAIT_VARIANTS` — default `'0,200,500'`. Comma-separated wait-ms variants.

### Critical reuse (no new code)

- `src/polygon_conditions.py` — `_classify_eligibility`, `_parse_conditions`, `_load_polygon_condition_rules`. Same canonical filter as flat-file ingestion uses for observable bars. Guaranteed parity.
- `src/flat_file_ingestion.py:_Bar` — the dataclass with `update_ohlc/update_volume` flags. We adapt to the streaming context but the OHLC/volume-only logic is identical.
- Existing `_run_polygon_ws` connection + auth + subscribe pattern — extend, don't rewrite.

### Verification

After deploy + 30 min of trading:
1. `SELECT symbol, source, COUNT(*) FROM live_bars_trades GROUP BY 1,2;` — should show rows for SPY, TSLA × three source labels at expected rates (10s bars: 6/min × 60 min = 360 per source per symbol per hour).
2. Direct DB query comparing trade-bar close to REST 1Sec close for a recent 15-min window — close-match should be ≥95% on the t_wait200 / t_wait500 variants. t_wait0 expected lower (misses in-flight trades).
3. Worker logs: no new errors, no backpressure indicators. Existing AM/A path keeps writing to `live_bars` exactly as before.
4. Frontend (Phase H.2 follow-up): Bars Comparison gets 3 new source options; user can visually verify wait-time variants.

### Rollback procedure

```bash
railway variables --service Worker --set POLYGON_TRADE_SHADOW_ENABLED=false
# Worker auto-redeploys; trade subscription disabled, builder inactive, ~3 min
```

Original alert path unchanged regardless of flag.

## Phase H.2 — Frontend trade-bar sources (next session)

Defer to next session unless time permits tonight after H.1 is verified.

- `frontend/src/charts/ParityBarComparison.tsx`: add 3 source keys (`t_wait0`, `t_wait200`, `t_wait500`)
- New hook `useTradeBars(symbol, start, end, tf_seconds, waitMs)` calling new endpoint
- New endpoint `/api/admin/parity/trade-bars` reads `live_bars_trades`

## Phase H.3 — Production cutover (gated, future session)

Only if shadow data confirms a variant achieves ≥95% close-match. Steps:
1. Pick the winning variant (most likely t_wait200 or t_wait500)
2. Change ralph_engine to write that variant's bars to `live_bars` `first_*` columns (replacing AM/A first-emission writes)
3. Alerts now fire on trade-built bars. Real-world latency: bar boundary + waitMs.
4. Validate via Bars Comparison before final flag flip.

## Critical Files to Read First

| Phase | Read first |
|---|---|
| H.1 | `src/ralph_engine.py:3098-3210` (Polygon WS subscription + message router), `src/flat_file_ingestion.py:236-310` (_Bar class + bar builder reference), `src/polygon_conditions.py` (whole file, ~180 lines), `src/migrations/live_bars_table.sql` (schema convention to mirror) |
| H.2 | `frontend/src/charts/ParityBarComparison.tsx`, `src/api/routers/admin_parity.py:283+` (rest-bars endpoint as reference) |
| H.3 | Re-read ralph_engine.py:1611-1700 (on_tick + bar-close monitor pipeline) to understand promotion points |

## Out of Scope (this plan)

- Volume-match optimization (still ~28-70% per Phase G measurements). Trade-channel approach should naturally fix close-match but volume requires separate work — fractional shares handling + vol-eligibility refinement. Document as future Phase H.4.
- Promotion to production alert source (Phase H.3, gated).
- Frontend visualization changes beyond adding dropdown options (Phase H.2).
- Multi-day storage retention (live_bars_trades will grow ~3x faster than live_bars due to multiple variants — revisit if disk/cost becomes a concern, otherwise add a retention purge job in Phase H.2).
