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

### Tier A — final state EOD 2026-04-28 (post-evening migration)

All 10 mirrors are now structurally clean (audit `_audit_strategy_migration.py` reports ALL_MIRRORED for every one). Configs were updated this session via `_migrate_mirrors_to_v2.py` (commit `29a8af1`):
- `bar_count_exit=4` → `time_exit_config={method: max_hold_bars, max_bars: 4}` on sids 135, 136, 144
- sid 144 entry trigger `ema_price_position_v2_default_cross_short_up` → `eppv4_default_cross_short_up` (Tier B v4 chosen — L1 1-bar lag closer to original C-type semantic)
- sid 144 confluence `1d-EMA_PRICE_POSITION-PSML` → `1d-EMA_PP_V4-PSML`
- `swing_123_default` confluence group re-enabled (was disabled, would have crashed worker per `feedback_disabled_groups_kill_worker.md`)

| Original | Mirror | Symbol/TF | Swaps | Stored trades | Parity status |
|---|---|---|---|---|---|
| sid 50 | **sid 136** | SPY/1Min | macd_line_v2, macd_histogram_v2 | populated | not yet run |
| sid 51 | **sid 135** | META/1Min | macd_line_v2, macd_histogram_v2, rvol_v2 | ✅ 208 trades | not yet run (this session) |
| sid 91 | **sid 137** | TSLA/5Min | vwap_v2, macd_histogram_v2 | populated | not yet run |
| sid 100 | **sid 138** | TSLA/5Min | macd_histogram_v2 | populated | not yet run |
| sid 102 | **sid 139** | TSLA/10Sec | rvol_v2 | populated | not yet run |
| sid 122 | **sid 140** | SPY/1Min | macd_histogram_v2 | populated | not yet run |
| sid 123 | **sid 141** | TSLL/1Min | vwap_v2 | populated | not yet run |
| sid 125 | **sid 142** | TSLA/10Sec | vwap_v2 | populated | not yet run |
| sid 127 | **sid 143** | TSLA/10Sec | macd_histogram_v2, vwap_v2 | populated | FAIL_LIVE_BLOCKED 0/122 (one-off run during contention; needs re-run) |
| sid 63 | **sid 144** | AMD/1Min | ema_stack_v2, eppv4 entry, EMA_PP_V4 confluence | ✅ 49 → 50 trades | PARTIAL 0/49 PRE-migration (replay_only=92). Re-run post-migration pending. |

### Tier B — not started

ema_price_position / ema_price_position_v2 references on: sid 88, 124, 134, 63 (entry trigger still). Each needs a v3-or-v4 decision.

### Tier C — not started (deferred indefinitely; treat as rebuild not migrate)

utbot_v2 references on: sid 59, 64, 66, 71, 114, 117, 131. Different state model in ut_bot_v4 — not a 1:1 swap.

## Live monitoring of mirrors (auto-discovered)

The worker monitors every strategy with a stamped `entry_trigger_confluence_id` regardless of whether stored_trades is populated. So as soon as a mirror is created, the worker warms it up and starts evaluating bars on it. Confirmed in worker logs 22:13:35 onward — sid 135-144 all initialized=True.

**Important:** Mirrors fire alerts in parallel with originals. They write to the alerts table. They do NOT fire webhooks because the new mirrors aren't in any portfolio (webhook dispatch requires `portfolio.webhook_group_id`). This gives us live-fired comparison data without duplicate webhook risk.

To turn off mirror live-firing if it becomes noisy: set `forward_testing=False` on each. To reactivate, flip it back.

## Verification protocol (post-evening update)

Auto-parity is now **user-triggered via the Run Parity button** (commit `1dc14bf`) — no longer auto-fires on refresh. To verify a mirror:

1. From My Strategies, select the mirror (or a batch).
2. Click **Run Parity** in the bulk-action bar. Backend queues a bg parity replay through `Semaphore(1)` per `718f692` so concurrent queues don't thrash the API.
3. The badge shows `⋯ Computing` until done. Reload page to see verdict.
4. Verdict colour-coding: green PASS / orange PARTIAL / red FAIL_LIVE_BLOCKED / FAIL_OVER_FIRES / muted NO_TRADES.
5. If PASS (≥95%), original can be retired.
6. If PARTIAL/FAIL, drill into Strategy Detail → Parity tab for per-trade diff.

For deeper diagnosis: the Parity tab still supports custom `last_n` + `forward_test_only` filters; the auto-stamped `parity_status` column uses defaults `last_n=200, forward_test_only=False`.

## Open issues

1. **~~Backtest cost is the migration bottleneck.~~ Withdrawn 2026-04-28 evening.** The "5-10 min per backtest" was concurrency overload, not single-call cost. Profiled isolated 180d × 1Min × META = 16s end-to-end. Real backtest bottleneck (when one matters) is per-bar `.iloc` in `unified_engine.process_bar` — ~6s of pure overhead per 180d. Future 10x project; not blocking.
2. **Bar cache via Supabase REST is a perf REGRESSION** for this workload. 30d direct Polygon = 0.9s, cache warm = 2.86s. Flag stays off. Correctness-clean (commits `ab49ebe` / `12e6701` / `574f04f`) but not enabled.
3. **forward_test_start preserved exactly** ✅ verified on sid 135 + spot-checked.
4. **Built-in + user-pack confluence groups must stay enabled** while strategies reference them. `swing_123_default` had been disabled this session — re-enabled in `29a8af1` migration. See `feedback_disabled_groups_kill_worker.md`.
5. **Strategy Detail "Loading indicators & heatmap" hang (RESOLVED).** Sid 127 took 10-20 min on TSLA/10Sec. Fixed in `2914f20` — vectorized state-column injection (~6s saved per chart load on 49k-bar strategies) + per-phase timing log. Verified: 5-6s end-to-end now.
6. **User packs lacking parity records (open).** Only `swing_123` (FAIL) + `ut_bot_v4` (FAIL) have records. The v2 packs the mirrors actually use have NO parity records. Run `POST /api/packs/builder/user-packs/{slug}/parity-test` on `macd_line_v2`, `macd_histogram_v2`, `vwap_v2`, `rvol_v2`, `ema_stack_v2`, `ema_pp_v3`, `ema_pp_v4`, `swing_123_test` next session.
7. **forward_trades_count flicker on strategy card (open).** Card shows `0` then `15` on consecutive page loads after a refresh. Likely race in `_augment_with_counts` with concurrent updates.

## Next session plan (post-button-split)

1. **Click Run Parity on the 10 Tier-A mirrors.** Should populate the verdict matrix in 5-15 min total (Semaphore(1) serializes so they take their natural sum, no thrashing).
2. **Run user-pack 4-quadrant parity** on the 8 v2 packs. Catches batch↔live drift at the pack level.
3. **Read the verdict matrix** + decide:
   - Green dominant → retire originals where mirror passes.
   - Orange/red dominant → fix the failing pack(s) at the pack level.
   - TF-correlated failures → engine-level bug.
4. **sid 144 specifically:** re-run Run Parity post-migration. The pre-migration result (0/49 matched, 92 replay_only) might have been a built-in template artifact — see if eppv4 cleared it.
5. **Retire originals** where parity passes (sid 50, 51, 91, 100, 102, 122, 123, 125, 127 — once their mirrors are PASS).
6. **Tier B walkthrough** for the non-mirror strategies that still reference `ema_price_position_v2` (sid 88, 124, 134, 66) — per-strategy v3 vs v4 decision.
7. **Tier C still deferred** (utbot_v2 → ut_bot_v4 needs rebuild not migrate; sid 59, 64, 66, 71, 114, 117, 131).
8. **Defer:** Layer 2 indicator pickle, `process_bar` vectorization, parity badge drilldown drawer, auto-clear parity_status on refresh, parity window sizing.
