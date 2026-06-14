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

## Human-in-loop findings (Kevin, 2026-06-13) + resolutions
- **Entries ALIGN.** Apparent 170930-vs-170940 gap = TV labels bars by OPEN time;
  a 10s bar `17:09:30` fills at its close 17:09:40 = our timestamp. Same event.
  Ours (fill time) is the more honest label. NOT a divergence.
- **Gate lagged ~1.5min on TV → FIXED.** The `[1]`+lookahead_off double-shift
  pushed TV's gate back an extra 2m bar. Dropped the `[1]` in B and C
  (lookahead_off alone = last closed bar = our semantics). Re-test expected to
  align the gate.
- **"Gate changes mid-candle / sub-minute" puzzle**: the gate VALUE only changes
  at 2m boundaries (even minutes), but it's forward-filled onto 10s bars, so the
  visual change appears on the FIRST PRESENT 10s bar after the boundary. On thin
  extended-hours tape, the :00/:10 bars right after a 2m close are often MISSING,
  so the change first shows at :20 etc. — looks sub-minute but isn't. It's a
  missing-bar rendering artifact, not real sub-minute gating. The CSV will
  confirm (check whether the 10s bars at the boundary exist).
- **Intrabar stop CONFIRMED possible on TV**: strategy.exit(stop=) fills mid-bar
  in the backtester → apples-to-apples with our L-type stop. Used in Checkpoint C.

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

## Bar-by-bar parity results (2026-06-13, Checkpoint B chart export vs engine)
CSV: docs/reference_images/AMEX_SPY, 10S (35).csv (961 bars, 18:49-23:59Z Fri).
Tool: src/_tv_bar_parity.py.
- (A) MATH parity (our ema_pp_v3 on TV's own closes vs TV EMA cols):
  EMA short Δ avg/max = 0.00000/0.00000; EMA mid = 0.00001/0.00099. → port math
  is exact. Entry: our raw cross=120, TV gated=43, 43⊂120, TV-only=0 (gate
  cleanly filters 120→43).
- (B) DATA parity:
  TV vs REST(backtest): 961 vs 962 bars, shared 961, TV-only=0, close Δ
    avg 0.00003 / max 0.0304. → backtest data ≈ TV bar-for-bar.
  TV vs CACHE(live): 961 vs 929, TV-only=33 (thin-tape live-WS gaps), ours-only=1,
    close Δ avg 0.0015 / max 0.28. → live feed slightly less complete than TV on
    thin tape (known class); settled REST matches.
NEXT: paste Checkpoint C (sid303_checkpointC.pine — full: gate-fix + ATR stop),
re-export BOTH chart-data (recheck gate timing post-[1]-fix) AND "List of Trades"
(trade-by-trade incl. stop exits) → run _tv_parity_compare.py for full trade parity.

## FULL-STRATEGY trade parity (2026-06-13, Checkpoint C vs sid 303 backtest)
TV "List of Trades" export (gate-fix + ATR stop), 2026-06-08→06-13 (~5 days,
TV gave more seconds history than expected). Tool: _tv_parity_compare.py.
- TV trades=673, our backtest=657. **MATCHED entries ±15s = 626 (~93%).**
- Entry price |Δ|: avg 0.0004 / max 0.04. → fills near-identical.
- Exit reason (stop vs signal cross) agree: 583/626 = **93%**.
- Mismatch breakdown (nearest-neighbor):
  TV-only 47 = 9 near(15-60s) + 15 loose(60-300s) + 23 absent(>300s).
  BT-only 31 = 9 near + 13 loose + 9 absent.
  → ~93% exact, ~3% same-trade-shifted (thin-tape entry-bar diffs), ~5%
    genuinely one-sided (thin-tape bar/gate/stop-cascade). Consistent with the
    data-axis divergence quantified earlier; NOT a port-fidelity problem
    (indicator math is exact, backtest data ≈ TV bar-for-bar).
VERDICT: sid 303 port is FAITHFUL. Ready to generalize → generator + tab.

## RTH-ONLY trade parity (2026-06-13) — the clean read
Same Checkpoint C export, filtered to 13:30-20:00 UTC (9:30-16:00 ET, liquid):
- TV entries=328, backtest=327. **MATCHED ±15s = 323 (98.5% of TV / 98.8% BT).**
- Entry price |Δ|: avg 0.0003 / max 0.0050.
- **ZERO genuinely-absent trades** (no >300s one-sided). All 5 TV-only / 4 BT-only
  are the SAME trade shifted a few seconds (tiny bar diffs).
CONCLUSION: full-window 93% vs RTH 98.6% → the entire residual was thin
extended-hours tape, not a logic/port issue. In liquid hours the TradingView
port is ESSENTIALLY EXACT. Strong external validation of our backtest.
NEXT (Kevin go-ahead): build the generator + export tab so any strategy ports.

## GENERATOR SHIPPED (2026-06-13)
- src/pine_generator.py: modular Pine v6 generator. EMITTERS registry
  (ema_pp_v3, ut_bot_v4); per-pack indicator/trigger/gate-state emitters +
  stop emitter + orchestrator. Regenerates the validated sid 303 Checkpoint C
  exactly (86 lines). Extension = add one PineEmitter subclass to EMITTERS.
- GET /api/strategies/{id}/pine-export → {ok, pine, error, caveats}; ok=false
  names the missing pack emitter. VERIFIED live on deployed API.
- Frontend: 'TradingView Export' tab on Strategy Detail (usePineExport hook,
  copy button, caveats, paste instructions). Awaiting Kevin visual QA.
- Pack coverage so far: ema_pp_v3 (triggers), ut_bot_v4 (gate states). Other
  packs raise a clear "add an emitter for <pack>" until implemented — the UI
  surfaces exactly which one to add next based on the strategies you try.
- VERIFY: pick a 1Min+ strategy in the app → TradingView Export tab → paste →
  deep TV backtest (months of history) → compare to our backtest.

## PACK EMITTER SWEEP (2026-06-13) — progressive log
Goal: Pine emitter for all 15 user packs. 3 dirs (rsi_zones, rsi_zones_3,
swing_123_test) are empty legacy (pycache only) — 15 real packs, 13 to add
(ema_pp_v3 + ut_bot_v4 already done).

### Wave 1 — EMA/MACD family — ✅ DONE
- ema_pp_v4: triggers (cross short/mid) + ordering states (P/S/M/L, 24 perms).
  Identical math to v3 (shared EmaPpEmitter, prefix-parameterized).
- ema_stack_v2: cross_bull/bear + mid_cross + SML/SLM/... ordering states.
- macd_line_v2: cross_bull/bear, zero crosses + M>S± / M<S± states.
- macd_histogram_v2: flip_pos/neg, momentum_shift + H+up/H+dn/H-up/H-dn states.
All hand-rolled to match seeding (EMA first-close; MACD line=0/signal=0 first
bar; signal=rorEma(line)). Unit-tested (triggers + gate states emit valid Pine);
303 regen no-regression.
VALIDATION NEEDED (Kevin): if you have a strategy using any of these as the
PRIMARY trigger or as a 2m gate, export it → paste in TV → check trade/gate
parity (RTH). MACD-primary: lines plot near 0 so not shown on price overlay
(intentional); signal still drives entries.
