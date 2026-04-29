# Tier B + Tier C Migration Plan — 2026-04-29

Per `/home/kevin/.claude/plans/eager-wiggling-twilight.md` Phase 3.
**Do NOT execute until Phase 1 (pack baseline) + Phase 2 (Tier-A
strategy parity) are complete.** Phase 3 needs a clean pack foundation;
running it before then risks declaring mirrors "done" while the underlying
packs are silently broken.

## Decision rules

**For Tier B (`ema_price_position_v2` → `ema_pp_v3` or `ema_pp_v4`):**
- `_ib` suffix in original trigger → **v3** (L0 intra-bar — matches the original `_ib` semantic)
- bare trigger (no `_ib`) → **v4** (L1 1-bar lag — closer to bar-close behavior)
- Confluence record `1d-EMA_PRICE_POSITION-...` → match the entry-trigger choice

**For Tier C (`utbot_v2` → `ut_bot_v4`):**
- ut_bot_v4 trigger_prefix is `utv4`. Group ID is `ut_bot_v4_default`. Trigger ID format: `utv4_default_{base}`.
- `utbot_v2_default_buy` → `utv4_default_bull_flip`
- `utbot_v2_default_sell` → `utv4_default_bear_flip`
- `utbot_v2_default_buy_ib` → `utv4_default_bull_flip` (no _ib variant in v4; L1 semantic via `_prev` columns is the closest equivalent — flag as a behaviour-change risk for sid 114)
- Confluence `{tf}-UTBOT-BULL` → `{tf}-UT_BOT_V4-BULL_TREND`
- Confluence `{tf}-UTBOT-BEAR` → `{tf}-UT_BOT_V4-BEAR_TREND`

**Other legacy templates encountered while migrating:**
- `macd_line_default_*` → `macd_line_v2_default_*`
- `{tf}-MACD_LINE-{state}` → `{tf}-MACD_LINE_V2-{state}`
- (etc. for any built-in template that has a `_v2` mirror)

## Per-strategy migration table

### Tier B

| sid | TF | sym | original entry | original exit | confluences | new entry | new exit | confluence swaps |
|---|---|---|---|---|---|---|---|---|
| 66  | 1Min  | AAPL | `ema_price_position_v2_default_cross_short_up` | `utbot_v2_default_sell` | `1d-MACD_LINE-M>S-`, `5m-UTBOT-BEAR` | `eppv4_default_cross_short_up` (v4 — bare) | `utv4_default_bear_flip` (Tier C) | `1d-MACD_LINE-M>S-`→`1d-MACD_LINE_V2-M>S-`; `5m-UTBOT-BEAR`→`5m-UT_BOT_V4-BEAR_TREND` |
| 88  | 10Sec | SPY  | `stochastic_oscillator_default_k_cross_above_d_oversold` (already user-pack) | `ema_price_position_v2_default_cross_short_down_ib` (_ib) | none | _no change_ | `eppv3_default_cross_short_down` (v3 — _ib semantic) | n/a |
| 124 | 1Min  | SPY  | `ema_price_position_v2_default_cross_short_up_ib` (_ib) | `ema_price_position_v2_default_cross_mid_down_ib` (_ib) | none | `eppv3_default_cross_short_up` (v3 — _ib) | `eppv3_default_cross_mid_down` (v3) | n/a |
| 134 | 10Sec | SPY  | `ema_price_position_v2_default_cross_short_up` (bare) | `ema_price_position_v2_default_cross_mid_down` (bare) | `15m-UT_BOT_V4-BULL_TREND` (already user-pack) | `eppv4_default_cross_short_up` (v4 — bare) | `eppv4_default_cross_mid_down` (v4) | n/a (already user-pack) |

### Tier C

| sid | TF | sym | original entry | original exit | confluences | new entry | new exit | confluence swaps |
|---|---|---|---|---|---|---|---|---|
| 59  | 1Min  | AAPL | `utbot_v2_default_buy` | `utbot_v2_default_sell` | `1d-MACD_LINE-M>S-`, `5m-UTBOT-BEAR` | `utv4_default_bull_flip` | `utv4_default_bear_flip` | `1d-MACD_LINE-M>S-`→`1d-MACD_LINE_V2-M>S-`; `5m-UTBOT-BEAR`→`5m-UT_BOT_V4-BEAR_TREND` |
| 64  | 1Min  | AAPL | `utbot_v2_default_buy` | `macd_line_default_cross_bear` | `1d-MACD_LINE-M>S-`, `1M-MACD_LINE-M<S-` | `utv4_default_bull_flip` | `macd_line_v2_default_cross_bear` | both `MACD_LINE`→`MACD_LINE_V2` |
| 66  | 1Min  | AAPL | (Tier B) | `utbot_v2_default_sell` | (covered above) | (Tier B) | `utv4_default_bear_flip` | (covered above) |
| 71  | 10Sec | SPY  | `utbot_v2_default_buy` | (none) | none | `utv4_default_bull_flip` | (none) | n/a |
| 114 | 10Sec | SPY  | `utbot_v2_default_buy_ib` (_ib) | `swing_123_default_bear_c3` (already user-pack) | none | `utv4_default_bull_flip` ⚠ behaviour change | (no change) | n/a |
| 117 | 10Sec | SPY  | `utbot_v2_default_buy` | (none) | none | `utv4_default_bull_flip` | (none) | n/a |
| 131 | 10Sec | SPY  | `utbot_v2_default_buy` | `swing_123_default_bear_c2` (already user-pack) | none | `utv4_default_bull_flip` | (no change) | n/a |

## ⚠ Behaviour-change risks to flag pre-execute

1. **sid 114** — original was `utbot_v2_default_buy_ib` (intra-bar trigger). v4 has no `_ib` variant; the L1 1-bar-lag semantic via `_prev` columns is the closest match but not identical. Mark this as "may behave differently" in the migration metadata and re-run parity carefully.

2. **All Tier C** — `utbot_v2` had continuous BULL/BEAR states; `ut_bot_v4` separates them into BULL_TREND/BEAR_TREND (continuous trend) AND BULL_FLIP/BEAR_FLIP (single-bar transition events). `_default_buy` mapped to `_bull_flip` (entry on the flip event), which is closer to the original `_default_buy` semantic. Confluence records `{tf}-UTBOT-BULL` map to `{tf}-UT_BOT_V4-BULL_TREND` (continuous gate, what the user expected from "currently bullish").

3. **No precedent for sid 88 partial exit-only swap** — sid 88's entry is already a user pack (`stochastic_oscillator`); only its exit needs swapping. The migration script should support partial swaps.

## Migration script approach

Extend `_migrate_mirrors_to_v2.py` pattern (idempotent dry-run / `--apply`). New script `_migrate_tier_bc.py`:

1. Per-strategy decisions baked in as a constant dict (no auto-detect; explicit choices = audit trail)
2. Dry-run prints the planned changes, no DB writes
3. `--apply` writes through `update_strategy_admin` (each strategy → one update with merged config)
4. **Creates new mirrors** (not modifies originals): clones each via `save_strategy_admin`, swaps triggers/confluence, preserves `forward_test_start`, names `[mirror {orig_sid}]`
5. New mirror sids: 145–148 (Tier B, 4) + 149–155 (Tier C, 7); sid 66 covered in Tier B + Tier C — handle as one mirror with both swaps applied
6. Re-run `_audit_strategy_migration.py` afterward to verify all new mirrors are ALL_MIRRORED

## Execution order (when Phase 3 is unblocked)

1. Read this doc + verify the per-strategy choices look right to Kevin
2. Dry-run: `../.venv/bin/python _migrate_tier_bc.py` — print planned changes, no DB writes
3. Apply: `../.venv/bin/python _migrate_tier_bc.py --apply`
4. Re-run audit, expect ALL_MIRRORED count → 16 + 11 = 27
5. Move to Phase 4 (full parity sweep on all 21 mirrors, including the new ones)

## Originals retained (not deleted)

Per the plan, originals stay in DB. They get `forward_testing=False, alert_tracking_enabled=False` only AFTER their mirrors PASS strategy-level parity (Phase 6). For Tier C strategies whose mirror underperforms (different state model = different trade set), the original may be the stronger strategy and should be kept active. Decide per-strategy after Phase 4 verdict matrix.
