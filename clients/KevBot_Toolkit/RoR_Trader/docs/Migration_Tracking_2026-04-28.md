# Strategy Migration Tracking — 2026-04-28

Goal: migrate Kevin's legacy strategies off built-in TF Confluence templates onto user-pack mirrors so we close the chronic backtest↔live parity gap. The mirrors are parity-clean by construction (their batch path is `for row in df: incremental.update_bar(...)`), so once a strategy is fully on mirrors it should never need a chart-engine debug session again.

## Tooling

- **Audit script** `src/_audit_strategy_migration.py` — categorizes every strategy as `MIRRORABLE / ALL_MIRRORED / BUILT_IN_NO_MIRROR`.
- **Migration script** `src/_migrate_tier_a.py` — creates duplicates with mirror swaps. Flags: `--execute`, `--no-backtest`, `--sids X,Y,Z`, `--exclude`. Preserves `forward_test_start` exactly. Naming: `[mirror {orig_sid}]`.
- **Parity service** `src/api/services/parity_service.py` — replays a strategy through StrategyMonitor + shadow engines and diffs against stored_trades. Surfaced as the **Parity** tab on Strategy Detail.
- **Backup branch** `dev-backup-pre-parity-check-2026-04-28` (pre-canary).

## Tier definitions

| Tier | Templates | Mirror semantics | Risk |
|---|---|---|---|
| A | macd_line, macd_histogram, rvol, vwap, ema_stack | Clean parity replacement (output sets identical) | Low |
| B | ema_price_position, ema_price_position_v2 | Fork — pick `ema_pp_v3` (L0) or `ema_pp_v4` (L1) per strategy | Medium |
| C | utbot, utbot_v2 | Family upgrade to `ut_bot_v4` (different states/triggers, not 1:1) | High |

## Current status (sid → mirror sid)

### Tier A — final state EOD 2026-04-28

| Original | Mirror | Symbol/TF | Swaps | Backtest status |
|---|---|---|---|---|
| sid 50 | **sid 136** | SPY/1Min | macd_line, macd_histogram | empty stored_trades |
| sid 51 | **sid 135** | META/1Min | macd_line, macd_histogram, rvol | ✅ 208 trades, 96-99% parity vs sid 51 |
| sid 91 | **sid 137** | TSLA/5Min | vwap, macd_histogram | empty stored_trades |
| sid 100 | **sid 138** | TSLA/5Min | macd_histogram | empty stored_trades |
| sid 102 | **sid 139** | TSLA/10Sec | rvol | empty stored_trades |
| sid 122 | **sid 140** | SPY/1Min | macd_histogram | empty stored_trades |
| sid 123 | **sid 141** | TSLL/1Min | vwap | empty stored_trades |
| sid 125 | **sid 142** | TSLA/10Sec | vwap | empty stored_trades |
| sid 127 | **sid 143** | TSLA/10Sec | macd_histogram, vwap | empty stored_trades |
| sid 63 | **sid 144** | AMD/1Min | ema_stack (partial — entry trigger still on ema_price_position_v2) | ✅ 49 trades, 11/11 forward-test matches (no regressions, +4 new trades from today's bars) |

### Tier B — not started

ema_price_position / ema_price_position_v2 references on: sid 88, 124, 134, 63 (entry trigger still). Each needs a v3-or-v4 decision.

### Tier C — not started (deferred indefinitely; treat as rebuild not migrate)

utbot_v2 references on: sid 59, 64, 66, 71, 114, 117, 131. Different state model in ut_bot_v4 — not a 1:1 swap.

## Live monitoring of mirrors (auto-discovered)

The worker monitors every strategy with a stamped `entry_trigger_confluence_id` regardless of whether stored_trades is populated. So as soon as a mirror is created, the worker warms it up and starts evaluating bars on it. Confirmed in worker logs 22:13:35 onward — sid 135-144 all initialized=True.

**Important:** Mirrors fire alerts in parallel with originals. They write to the alerts table. They do NOT fire webhooks because the new mirrors aren't in any portfolio (webhook dispatch requires `portfolio.webhook_group_id`). This gives us live-fired comparison data without duplicate webhook risk.

To turn off mirror live-firing if it becomes noisy: set `forward_testing=False` on each. To reactivate, flip it back.

## Verification protocol

For each mirror (after backtest runs):
1. Open Strategy Detail → Parity tab, set `last_n=25, forward_test_only=true`
2. Read parity_score; expect ≥95% on Tier A
3. If PASS, original can be retired (delete or archive)
4. If <95%, investigate via the Reason breakdown column

## Open issues

1. **~~Backtest cost is the migration bottleneck.~~ Withdrawn 2026-04-28.** The "5-10 min per backtest" figure was conflating *single backtest* cost with the queued-up cost of running heavy backtests *concurrently* with mirror creation, data updates, and parity checks. Profiling a single 180d × 1Min backtest in isolation produced **~16 seconds** end-to-end (Stage 1 data+indicators ≈ 8s, Stage 2 unified engine ≈ 8s). The real bottleneck during the original migration session was concurrency overload, not single-call cost — the cat-shutdown reset was actually a clean break.
   - **Implication:** persistent bar cache (Layer 1) is *correctness-clean* (Supabase Postgres + delta-fetch + historical-tail backfill, commits `ab49ebe` / `12e6701` / `574f04f`) but is **not enabled** and not needed for the migration. Polygon REST is faster than Supabase REST for raw OHLCV at this scale. Layer 2 (indicator state pickle) is also de-prioritized; the indicator pipeline runs in a few seconds, so the win is small.
   - **The actual 10x opportunity** (if needed later) is vectorizing `unified_engine.process_bar` — currently 47k bars × `df.iloc[i]` accounts for ~6s of every 180d backtest. Filed for future, not blocking.
2. **forward_test_start preserved exactly** ✅ verified on sid 135 + spot-checked on others.
3. **Built-in confluence groups stay enabled** until every strategy is migrated. Disabling early kills the worker (see `feedback_disabled_groups_kill_worker.md`).
4. **Strategy Detail "Loading indicators & heatmap data..." hangs (open).** Sid 127 (TSLA/10Sec) reproducibly takes 10-20 min to load chart-data, sometimes never completing. Likely contributors: 49k bar payload over single API process, Polygon rate-limiting during concurrent migration spam, possible 50s+ pandas `.copy()` on cached DataFrame. Diagnosis blocked on lack of per-phase server timing — adding instrumentation in this session.

## Revised next session plan (post-profiling)

1. ~~Ship Layer 1 of persistent bar cache.~~ ✅ Shipped + correctness fixed; flag stays off.
2. **Diagnose the chart-data hang on sid 127 first** — perf is more user-visible than the migration backlog.
3. **Bulk-refresh the 9 Tier A mirrors** sequentially via the existing `recompute_and_persist_stored_trades` path. Expected total time: ~2-3 minutes (9 × 16s), well under the original 15 min target.
4. Run parity check on each (Strategy Detail → Parity tab, `last_n=25, forward_test_only=true`).
5. Retire originals where parity passes (≥95% on Tier A).
6. Talk through Tier B (v3-vs-v4 per strategy).
7. **Defer:** Layer 2 indicator pickle, vectorization of `process_bar`. Both are real but neither blocks anything user-visible right now.
