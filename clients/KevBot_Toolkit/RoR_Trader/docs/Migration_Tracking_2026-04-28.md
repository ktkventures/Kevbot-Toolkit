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

### Tier A — duplicates created, backtests pending

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
| sid 63 | **sid 144** | AMD/1Min | ema_stack (partial — entry trigger still on ema_price_position_v2) | empty stored_trades |

### Tier B — not started

ema_price_position / ema_price_position_v2 references on: sid 88, 124, 134, 63 (entry trigger still). Each needs a v3-or-v4 decision.

### Tier C — not started (deferred indefinitely; treat as rebuild not migrate)

utbot_v2 references on: sid 59, 64, 66, 71, 114, 117, 131. Different state model in ut_bot_v4 — not a 1:1 swap.

## Verification protocol

For each mirror (after backtest runs):
1. Open Strategy Detail → Parity tab, set `last_n=25, forward_test_only=true`
2. Read parity_score; expect ≥95% on Tier A
3. If PASS, original can be retired (delete or archive)
4. If <95%, investigate via the Reason breakdown column

## Open issues

1. **Backtest cost is the migration bottleneck.** Each 1Min × 180d backtest is 5-10 minutes. With 9 mirrors pending, that's ~45-90 minutes sequentially. **Persistent bar cache (project_persistent_bar_cache.md) would cut this by 70-90% on cold runs and ~99% on warm runs.** Schedule this BEFORE bulk-refreshing the mirrors.
2. **forward_test_start preserved exactly** ✅ verified on sid 135 + spot-checked on others.
3. **Built-in confluence groups stay enabled** until every strategy is migrated. Disabling early kills the worker (see `feedback_disabled_groups_kill_worker.md`).

## Next session plan

1. Ship Layer 1 of persistent bar cache (parquet on disk + Polygon delta fetch).
2. Verify bulk-refresh of the 9 Tier A mirrors completes in <15 min total.
3. Run parity check on each; record results in this doc.
4. Retire originals where parity passes.
5. Talk through Tier B (v3-vs-v4 per strategy).
