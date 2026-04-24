# Plan — Trades Table Migration (2026-04-24)

## Goal

Move `strategies.stored_trades` (JSONB, up to 2.5 MB per strategy) into a normalized `trades` table. Fixes:
- `_persist_algo_trade` statement-timeout cascades (direct cause of today's 11:48-11:56 MT alert-history lag)
- Exit-webhook latency (~0.5-2s/exit improvement under load)
- Supabase baseline read pressure (strategy detail page pulls multi-MB on every click)

## Design — Schema (Option B: hot columns + JSONB tail)

```sql
CREATE TABLE trades (
  -- Identity / joining
  id              BIGSERIAL PRIMARY KEY,
  strategy_id     INTEGER NOT NULL,
  user_id         UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  entry_alert_id  BIGINT,          -- join to alerts table when trade was live/matched
  exit_alert_id   BIGINT,

  -- Time (4-timestamp spec)
  entry_trigger_ts TIMESTAMPTZ,
  entry_fill_ts    TIMESTAMPTZ,
  exit_trigger_ts  TIMESTAMPTZ,
  exit_fill_ts     TIMESTAMPTZ,

  -- Core trade economics (heavily queried)
  entry_price      NUMERIC,
  exit_price       NUMERIC,
  r_multiple       NUMERIC,
  dollar_pnl       NUMERIC,
  executed_quantity INTEGER,

  -- Routing / classification
  direction        TEXT,
  exec_type        TEXT,
  exit_reason      TEXT,
  data_source      TEXT,          -- 'backtest' | 'forward_test' | 'matched' | 'live' | 'raw_alerts'

  -- Long-tail fields stored as JSONB `data`:
  --   stop_price, initial_stop_price, target_price, risk, pnl, win,
  --   hold_duration_s, bars_held, entry_trigger, exit_trigger,
  --   stop_exec_type, target_exec_type, behavior, confluence_records,
  --   planned_quantity, rpt_quantity, bp_quantity, buying_power_used,
  --   risk_per_trade, entry_slippage_r, exit_slippage_r,
  --   theoretical_entry, theoretical_exit, entry_stop_price
  data             JSONB DEFAULT '{}'::jsonb,

  -- Housekeeping
  created_at       TIMESTAMPTZ DEFAULT NOW()
);

-- Foreign key to strategies (clean up trades when strategy deleted)
ALTER TABLE trades
  ADD CONSTRAINT trades_strategy_id_fkey
  FOREIGN KEY (strategy_id) REFERENCES strategies(id) ON DELETE CASCADE;

-- Indexes
CREATE INDEX trades_strategy_exit_idx   ON trades (strategy_id, exit_fill_ts DESC);
CREATE INDEX trades_strategy_entry_idx  ON trades (strategy_id, entry_fill_ts DESC);
CREATE INDEX trades_user_id_idx         ON trades (user_id);
CREATE INDEX trades_entry_alert_idx     ON trades (entry_alert_id) WHERE entry_alert_id IS NOT NULL;
CREATE INDEX trades_exit_alert_idx      ON trades (exit_alert_id)  WHERE exit_alert_id IS NOT NULL;

-- RLS (same pattern as webhook_groups / strategies)
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
CREATE POLICY trades_select_own ON trades FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY trades_insert_own ON trades FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY trades_update_own ON trades FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY trades_delete_own ON trades FOR DELETE USING (auth.uid() = user_id);
```

## Strategy — Feature-flagged cutover

Env var: `USE_TRADES_TABLE` (default `false`).

- **OFF** (default): everything works like today — `stored_trades` JSONB is canonical.
- **ON**: writes go to `trades` table only; reads come from `trades` table. JSONB column stops receiving new writes.
- **Flip back safely**: old JSONB is still intact (historical). Trades written during ON window need backtest recompute to be restored to JSONB — recoverable, not data loss.

## Four commits

### Commit 1 — Migration SQL (Kevin runs it)
File: `src/migrations/phase40_trades_table.sql`

1. `CREATE TABLE trades` + all indexes + RLS policies (above)
2. Backfill: iterate `strategies.stored_trades` and INSERT rows into `trades`. Idempotent via `ON CONFLICT DO NOTHING` keyed on `(strategy_id, entry_fill_ts, exit_fill_ts)` — run multiple times safely.
3. Verification SELECT: counts match.

**Risk**: zero runtime risk. Pure data duplication. App behavior unchanged.

### Commit 2 — Backend branching + env var
Files: `src/db.py`, `src/services.py`, `src/worker.py`, `src/api/services/forward_test_service.py`, `src/mass_builder.py`, `src/api/routers/strategies.py`, `src/api/routers/portfolios.py`

Strategy:

1. **New helper module** `src/trades_store.py`:
   - `_flag_on() -> bool` — reads `USE_TRADES_TABLE` env var
   - `load_trades_for_strategy(strategy_id, user_id=None) -> list[dict]` — returns trade records in the same shape as legacy `stored_trades` entries
   - `insert_trade(strategy_id, user_id, trade_record)` — single row INSERT
   - `replace_trades_for_strategy(strategy_id, user_id, trade_records)` — atomic DELETE + bulk INSERT (for recompute path)
   - `hydrate_strategy_trades(strategy: dict) -> dict` — given a strategy dict, populate `strategy['stored_trades']` from the new table. Called only by detail/hydrate endpoints (list endpoints stay lean).
   - Each function branches on `_flag_on()`; when flag OFF, defers to existing JSONB paths.

2. **Write path branching** — every `update_strategy_admin(..., {'stored_trades': ...})` or equivalent call goes through `trades_store.replace_trades_for_strategy` when flag ON:
   - `worker.py:_persist_algo_trade` → INSERT single trade row (append-only, not rewrite)
   - `forward_test_service.recompute_and_persist_stored_trades` → DELETE then bulk INSERT + still write kpis + equity_curve_data via `update_strategy_admin` (those stay as columns)
   - `mass_builder.py` backtest-save paths → same replacement pattern

3. **Read path branching** — `services.trades_df_from_stored(strategy)` becomes `services.get_strategy_trades_df(strategy)`:
   - Flag OFF: read `strategy.stored_trades` as today
   - Flag ON: call `trades_store.load_trades_for_strategy` and convert to DataFrame
   - All ~30 read sites use this helper (many already do)

4. **API response shape preservation** — Detail/hydrate/trades endpoints continue returning `strategy.stored_trades` array. When flag ON, `_row_to_strategy` (or the endpoint handler) calls `hydrate_strategy_trades()` before returning. **Frontend requires zero changes.**

5. **Backward compat for double-restore safety**: when a write happens while flag is ON, ALSO append to `stored_trades` JSONB for a transition period? **No** — this would defeat the whole perf win. Skip double-write. Accept that flipping OFF requires a recompute to reseed JSONB.

### Commit 3 — Kevin flips `USE_TRADES_TABLE=true` on Railway

Not a code change — just a config flip + restart.

Validation:
- Fire an entry/exit cycle; confirm webhook fires as expected
- Load Strategy Detail for strategy 114 (~3000 trades); confirm trades table renders correctly
- Confirm Supabase Query Performance shows big drop in `_persist_algo_trade` latency
- Open a portfolio with mixed strategies; confirm Trade History merges correctly

If anything's off, flip `USE_TRADES_TABLE=false` on Railway and restart. Rollback time: ~60 seconds. No data loss.

### Commit 4 — Cleanup (later, after 24-48 hours of confidence)

- Remove the branching code — keep only the new-table path
- Remove `trades_store._flag_on()` and all branches
- Leave `strategies.stored_trades` column in place for one more week (belt-and-braces)
- Eventually: `ALTER TABLE strategies DROP COLUMN stored_trades` (weeks from now)

## What frontend sees

**Nothing changes on the frontend.** Every API response continues to include `strategy.stored_trades` array in the same shape. Only the backend storage changes.

Why this is safe: hydrate-at-response-boundary pattern. The frontend's expected DTO is preserved. We can refactor frontend to use a dedicated `/api/strategies/{id}/trades?page=N` paginated endpoint later — that's an optimization, not a correctness requirement.

## Files touched (Commit 2)

| File | Change |
|---|---|
| `src/trades_store.py` | NEW — flag-gated trades CRUD + hydration helper |
| `src/db.py` | Add `load_trades_db`, `insert_trade_db`, `delete_trades_for_strategy_db`, `get_strategy_by_id_with_trades_admin` |
| `src/services.py` | `trades_df_from_stored` delegates to `trades_store.get_strategy_trades_df` |
| `src/worker.py` | `_persist_algo_trade` branches through `trades_store.insert_trade` |
| `src/api/services/forward_test_service.py` | `recompute_and_persist_stored_trades` branches through `trades_store.replace_trades_for_strategy` |
| `src/mass_builder.py` | Save paths branch similarly |
| `src/api/routers/strategies.py` | Detail/hydrate/trades handlers call `hydrate_strategy_trades` when flag ON |
| `src/api/routers/portfolios.py` | Same for /trades endpoint |
| `src/app.py` | Legacy Streamlit app — branch write paths (or leave if Streamlit isn't actively used) |

## Backfill details (Commit 1 SQL)

```sql
-- One-shot: unnest every strategies.stored_trades row into the trades table.
-- Idempotent via ON CONFLICT (covered by a unique index on
-- (strategy_id, entry_fill_ts, exit_fill_ts) so re-runs are safe).
CREATE UNIQUE INDEX IF NOT EXISTS trades_dedupe_idx
  ON trades (strategy_id, entry_fill_ts, exit_fill_ts);

INSERT INTO trades (
  strategy_id, user_id,
  entry_alert_id, exit_alert_id,
  entry_trigger_ts, entry_fill_ts, exit_trigger_ts, exit_fill_ts,
  entry_price, exit_price,
  r_multiple, dollar_pnl, executed_quantity,
  direction, exec_type, exit_reason, data_source,
  data
)
SELECT
  s.id AS strategy_id,
  s.user_id,
  (t->>'entry_alert_id')::BIGINT,
  (t->>'exit_alert_id')::BIGINT,
  (t->>'entry_trigger_ts')::TIMESTAMPTZ,
  COALESCE((t->>'entry_fill_ts')::TIMESTAMPTZ, (t->>'entry_time')::TIMESTAMPTZ),
  (t->>'exit_trigger_ts')::TIMESTAMPTZ,
  COALESCE((t->>'exit_fill_ts')::TIMESTAMPTZ,  (t->>'exit_time')::TIMESTAMPTZ),
  NULLIF(t->>'entry_price','')::NUMERIC,
  NULLIF(t->>'exit_price','')::NUMERIC,
  NULLIF(t->>'r_multiple','')::NUMERIC,
  NULLIF(t->>'dollar_pnl','')::NUMERIC,
  NULLIF(t->>'executed_quantity','')::INTEGER,
  t->>'direction',
  t->>'exec_type',
  t->>'exit_reason',
  COALESCE(t->>'data_source', 'backtest'),
  t - '{entry_alert_id,exit_alert_id,entry_trigger_ts,entry_fill_ts,exit_trigger_ts,exit_fill_ts,entry_price,exit_price,r_multiple,dollar_pnl,executed_quantity,direction,exec_type,exit_reason,data_source,entry_time,exit_time}'::text[]
FROM strategies s,
     LATERAL jsonb_array_elements(COALESCE(s.stored_trades, '[]'::jsonb)) AS t
ON CONFLICT (strategy_id, entry_fill_ts, exit_fill_ts) DO NOTHING;

-- Verify
SELECT
  (SELECT SUM(jsonb_array_length(COALESCE(stored_trades, '[]'::jsonb))) FROM strategies) AS jsonb_trade_count,
  (SELECT COUNT(*) FROM trades) AS table_trade_count;
```

Counts should be very close. Small mismatches expected for trades missing both `entry_fill_ts` and `entry_time` — those skip the dedupe index and would be rejected. We'll log those cases.

## Test plan (not exhaustive — validation before the flag flips)

Before flipping `USE_TRADES_TABLE=true`:

1. **Backfill count check** — the SQL above returns `jsonb_trade_count ≈ table_trade_count`. If off by more than 1%, investigate.
2. **Round-trip single strategy** — set flag ON, open Strategy Detail for a small strategy (< 10 trades), confirm trades table renders same data as before. Set flag OFF, confirm same.
3. **Fire one trade under flag ON** — let engine fire one live entry+exit. Confirm: trade appears in new `trades` table + webhook delivers + Strategy Detail shows it.
4. **Portfolio trade history under flag ON** — open portfolio with multiple strategies, confirm merged view matches Alert History.

## Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Flipping flag ON exposes a bug in read path | MEDIUM | Flip off, recompute |
| Backfill misses some trades (edge-case timestamp shapes) | LOW-MED | Verify counts match; re-run backfill with looser conversion |
| Mass Builder generates tons of trades during flag transition window | LOW | Kick off Mass Builder only AFTER flag flip confirmed |
| Frontend breakage from DTO drift | VERY LOW | Hydrate-at-response pattern keeps DTO identical |
| Lost write during flag flip | LOW | Flag is read per-call; flip takes effect immediately after next request |
| Stale `kpis` / `equity_curve_data` columns | LOW | Both still written via `update_strategy_admin` in recompute path (unchanged) |

## Backup
Branch `dev-backup-pre-trades-migration-2026-04-24` at commit `896a163` is pushed. Safe revert point.

## Timeline estimate
- Commit 1 (migration SQL): ~45 min
- Commit 2 (code branching): ~2-3 hours
- Manual verification + flag flip: ~30 min
- Commit 4 cleanup: separate session

Total: ~4 hours for the complete migration with validation. Kevin has the runway today.
