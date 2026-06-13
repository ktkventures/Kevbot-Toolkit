# TradingView Export — Design Spec (started 2026-06-13)

## Goal
A modular "TradingView Export" tab on Strategy Detail. For ANY strategy, generate
a **Pine Script v6 `strategy()`** that faithfully recreates it (indicators,
cross-TF confluence + heatmap visual, entry/exit) so a user can paste it into
TradingView, see a familiar chart, run TradingView's trusted Strategy Tester,
and fire **alerts/webhooks** for live execution. Purposes: marketing trust, a
live-trading path independent of our in-app live model, external backtest
validation.

## Decisions (Kevin, 2026-06-13)
- Output = Pine **`strategy()`** (Strategy Tester + alerts).
- v1 scope = **C-type bar-close, RTH** first (isolate fill risk).
- First target = **sid 303** (SPY 10Sec + 2m UT_BOT_V4 gate).
- Validation = **human-in-loop** (I generate Pine; Kevin pastes into TV, shares
  Strategy Tester results; I iterate). No TV automation in v1 (ToS/fragility).

## The port mechanism (reality)
- No official TradingView API to upload/run strategies. Pine is pasted into the
  Pine Editor. Live integration = Pine `alert()` → webhook.
- Validation can't be fully headless (no results API; UI automation is fragile/
  ToS-risky). Loop = generate → paste → compare trade lists.

## Architecture (modular, mirrors our pack system)
- Backend **Pine generator**: strategy config → `.pine` string, built from
  composable emitters:
  - per-pack indicator emitter (math + seeding) — one per user pack
  - per-pack interpreter/state emitter
  - trigger emitter (entry/exit)
  - confluence-gate emitter (request.security, lookahead_off, confirmed bars)
  - heatmap-visual emitter (Pine table/bgcolor analog)
  - stop/target + alert emitters
- Key insight: cost is **per-pack**, amortized — once a pack has a Pine emitter,
  every strategy using it ports for free. Generator is strategy-agnostic.
- Then: backend endpoint + Strategy Detail "TradingView Export" tab (code box +
  copy + paste instructions + seconds-history caveat + parity checklist).

## sid 303 reverse-engineered (source of truth for the port)
From `user_packs/ema_pp_v3` + `user_packs/ut_bot_v4` + 303 config:
- **EMA Price-Position v3** (defaults short=8, mid=21, long=50):
  `ema = first-bar close; else a*close+(1-a)*ema, a=2/(p+1)`. **Seeds at first
  close — NOT Pine's ta.ema (SMA seed). Must hand-roll.**
  - entry `eppv3_cross_short_up` = `close>emaShort and prevClose<=prevEmaShort`
    ≡ Pine `ta.crossover(close, emaShort)` with matched seeding.
  - exit `eppv3_cross_mid_down` = `close<emaMid and prevClose>=prevEmaMid`
    ≡ `ta.crossunder(close, emaMid)`.
- **UT Bot V4** (defaults atr_period=10, key_value=1.0, Wilder ATR seeded at
  first TR): classic ATR trailing-stop, 4-case ratchet. `BULL_TREND = close >
  trailing_stop` (used by the 2m gate). Hand-roll TR/ATR to match seeding.
- **2m gate** `2m-UT_BOT_V4-BULL_TREND`: gates on the PREVIOUS CLOSED 2m bar
  (our cross-TF look-ahead fix). Pine: `request.security(sym, "2",
  isBullTrend[1], lookahead=barmerge.lookahead_off)`. ← #1 divergence risk.
- **Stop** `stop_config = {method:atr, atr_mult:1.5, exec_type:L}`. No target.
  L-type (intrabar) — the one non-C element; Checkpoint C.
- Session = Extended Hours. risk/trade $100, start $10k.

## Checkpoints (within 303) — sequence A→B→C (Kevin, 2026-06-13)
- **A** (DONE, `sid303_checkpointA.pine`): EMAs + C-type entry/exit, no gate, no
  stop. Proves indicator/cross/seeding + bar-close fill parity in isolation.
- **B** (DONE, `sid303_checkpointB.pine`): + 2m UT_BOT_V4 BULL_TREND gate via
  request.security(lookahead_off, prev-bar). Fidelity catch: BULL_TREND =
  (close>stop AND not bullFlip) — the flip-up bar is BULL_FLIP, not BULL_TREND.
  Gate-state shown as bgcolor + corner table (first heatmap analog).
- **C** (NEXT, after A/B validated in TV): + ATR stop. CONFIRMED semantics
  (triggers.py:75-118): **fixed-at-entry**, `stop = entry − atr_mult×ATR` at the
  entry bar (NOT trailing), atr_mult=1.5, exec_type L (resting/intrabar fill).
  Pine: strategy.exit(stop=...). Hand-roll ATR (Wilder) to match our 'atr'
  column seeding. OPEN: confirm ATR period of our 'atr' column (likely 14).
- Then: generalize into the generator + endpoint + tab (#15/#16/#17).

## Validation status
A and B are pasteable; awaiting human-in-loop TV test (paste → compare entry/
exit bars + counts vs sid 303 backtest on a recent day). The #1 thing to watch
in B: whether entries are shifted one 2m bar (if so, drop the [1] in the gate's
request.security — lookahead_off already returns last-confirmed).

## Divergence risks (honest)
1. Cross-TF repaint/look-ahead (request.security) — must use lookahead_off +
   confirmed/prev bar. Same battle as our cross-TF parity.
2. Custom-pack fidelity — each pack hand-reimplemented; seeding details matter.
3. Intrabar/L-type fills in Pine's backtester (bar magnifier helps on Premium).
4. **Seconds history**: TV keeps only days of seconds bars → 10Sec strategies
   backtest shallow on TV. Applies to all our 10Sec strategies. A 1Min+
   strategy is a better flagship for DEEP TV backtests; generator is reusable.
5. Pine limits: request.security count, script size, max bars back (watch when
   many conditions/TFs).

## Open items to confirm before B/C
- Confirm 303 uses pack-default params (8/21/50; atr10/key1.0) vs confluence-
  group overrides.
- ATR stop: fixed-at-entry distance vs trailing? (read stop method impl).
- Verify Premium exposes 10S bars + how much history.
