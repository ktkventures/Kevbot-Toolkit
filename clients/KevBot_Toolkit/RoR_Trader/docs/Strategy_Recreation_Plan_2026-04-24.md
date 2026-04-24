# Strategy Recreation Plan — 2026-04-24

Generated from a full DB audit of strategies with **forward algo trades but zero live alerts**. These are strategies where the unified engine's backtest path produces trades, but the live-bar engine never fires alerts — most likely cause is legacy trigger names or stale confluence-pack references that the live path doesn't resolve.

## How to use this doc

1. Open Strategy Builder

2. For each strategy below, configure with the listed parameters — **picking from the current trigger/pack pickers** so any renamed/removed items get a modern equivalent

3. Save with name `Recreated <ID> — <original name>` so you can match new vs old

4. Once the recreated strategy starts firing alerts (give it ~30 min), delete the old strategy

5. If a portfolio referenced the old strategy, swap to the new strategy id on the portfolio's Strategies tab


## Watch-outs

- Some triggers below use the **legacy `_default_` naming convention** (e.g. `macd_line_default_cross_bull`). Modern equivalents likely use `_v2_` (e.g. `macd_line_v2_cross_bull`) or a `_pack_` ref. Pick the closest match in the picker.

- `swing_123_test_*` triggers are confirmed working on strategy 117 — those are current.

- `utbot_v2_*` triggers are also current and working (strategies 71/114/117 use them).

- If no modern equivalent exists, pick the next-best trigger that captures the original strategy's thesis. Note in the new strategy's name (e.g. `Recreated 50 — MACD bull cross [updated trigger]`).


---


## Strategy 50 — SPY LONG - Mass #11

- **Symbol:** SPY
- **Direction:** LONG
- **Timeframe:** 1Min
- **Session:** RTH
- **Created:** 2026-03-17T23:38:37.742669+00:00
- **Original FW Start:** 2026-03-17T23:38:37.742672+00:00

**Triggers:**
- Entry: `macd_line_default_cross_bull`
- Exit: (none — signal-only? check original)

**Risk packs:**
- Stop: `(unset)` (method: swing, padding: 0.03, lookback: 5)
- Target: `(unset)` (signal exit only)
- Time exit: `(unset)` (method: --, max_bars: --)

**Confluences:** 15m-MACD_HISTOGRAM-H+dn, 1M-MACD_LINE-M>S-

**Data lookback:** 180 days, mode: `days`

**Recreation checklist:**
- [ ] New strategy created and named `Recreated 50 — SPY LONG - Mass #11`
- [ ] Verified entry trigger picker found the same/equivalent
- [ ] Saved + waited ~30 min for first signal
- [ ] First alert fired (check alert count on Strategies list page)
- [ ] Deleted strategy 50


## Strategy 51 — META LONG - Mass #13

- **Symbol:** META
- **Direction:** LONG
- **Timeframe:** 1Min
- **Session:** RTH
- **Created:** 2026-03-17T23:39:05.976499+00:00
- **Original FW Start:** 2026-03-17T23:39:05.976502+00:00

**Triggers:**
- Entry: `macd_line_default_cross_bull`
- Exit: (none — signal-only? check original)

**Risk packs:**
- Stop: `(unset)` (method: swing, padding: 0.03, lookback: 5)
- Target: `(unset)` (signal exit only)
- Time exit: `(unset)` (method: --, max_bars: --)

**Confluences:** 15m-MACD_HISTOGRAM-H+dn, 15m-RVOL-NORMAL

**Data lookback:** 180 days, mode: `days`

**Recreation checklist:**
- [ ] New strategy created and named `Recreated 51 — META LONG - Mass #13`
- [ ] Verified entry trigger picker found the same/equivalent
- [ ] Saved + waited ~30 min for first signal
- [ ] First alert fired (check alert count on Strategies list page)
- [ ] Deleted strategy 51


## Strategy 59 — AAPL LONG - Mass #5

- **Symbol:** AAPL
- **Direction:** LONG
- **Timeframe:** 1Min
- **Session:** RTH
- **Created:** 2026-03-18T20:23:58.993841+00:00
- **Original FW Start:** 2026-03-18T20:23:58.993843+00:00

**Triggers:**
- Entry: `utbot_v2_default_buy`
- Exit: `utbot_v2_default_sell`

**Risk packs:**
- Stop: `(unset)` (method: swing, padding: 0.03, lookback: 5)
- Target: `(unset)` (signal exit only)
- Time exit: `(unset)` (method: --, max_bars: --)

**Confluences:** 1d-MACD_LINE-M>S-, 5m-UTBOT-BEAR

**Data lookback:** 90 days, mode: `days`

**Recreation checklist:**
- [ ] New strategy created and named `Recreated 59 — AAPL LONG - Mass #5`
- [ ] Verified entry trigger picker found the same/equivalent
- [ ] Saved + waited ~30 min for first signal
- [ ] First alert fired (check alert count on Strategies list page)
- [ ] Deleted strategy 59


## Strategy 63 — AMD LONG - Mass #14

- **Symbol:** AMD
- **Direction:** LONG
- **Timeframe:** 1Min
- **Session:** RTH
- **Created:** 2026-03-18T20:27:08.353493+00:00
- **Original FW Start:** 2026-03-18T20:27:08.353495+00:00

**Triggers:**
- Entry: `ema_price_position_v2_default_cross_short_up`
- Exit: (none — signal-only? check original)

**Risk packs:**
- Stop: `(unset)` (method: swing, padding: 0.03, lookback: 5)
- Target: `(unset)` (signal exit only)
- Time exit: `(unset)` (method: --, max_bars: --)

**Confluences:** 15m-EMA_STACK-MLS, 1d-EMA_PRICE_POSITION-PSML

**Data lookback:** 90 days, mode: `days`

**Recreation checklist:**
- [ ] New strategy created and named `Recreated 63 — AMD LONG - Mass #14`
- [ ] Verified entry trigger picker found the same/equivalent
- [ ] Saved + waited ~30 min for first signal
- [ ] First alert fired (check alert count on Strategies list page)
- [ ] Deleted strategy 63


## Strategy 64 — AAPL LONG - Mass #16

- **Symbol:** AAPL
- **Direction:** LONG
- **Timeframe:** 1Min
- **Session:** RTH
- **Created:** 2026-03-18T20:29:51.628492+00:00
- **Original FW Start:** 2026-03-18T20:29:51.628495+00:00

**Triggers:**
- Entry: `utbot_v2_default_buy`
- Exit: `macd_line_default_cross_bear`

**Risk packs:**
- Stop: `(unset)` (method: swing, padding: 0.03, lookback: 5)
- Target: `(unset)` (signal exit only)
- Time exit: `(unset)` (method: --, max_bars: --)

**Confluences:** 1d-MACD_LINE-M>S-, 1M-MACD_LINE-M<S-

**Data lookback:** 90 days, mode: `days`

**Recreation checklist:**
- [ ] New strategy created and named `Recreated 64 — AAPL LONG - Mass #16`
- [ ] Verified entry trigger picker found the same/equivalent
- [ ] Saved + waited ~30 min for first signal
- [ ] First alert fired (check alert count on Strategies list page)
- [ ] Deleted strategy 64


## Strategy 66 — AAPL LONG - Mass #23

- **Symbol:** AAPL
- **Direction:** LONG
- **Timeframe:** 1Min
- **Session:** RTH
- **Created:** 2026-03-18T20:30:38.227769+00:00
- **Original FW Start:** 2026-03-18T20:30:38.227771+00:00

**Triggers:**
- Entry: `ema_price_position_v2_default_cross_short_up`
- Exit: `utbot_v2_default_sell`

**Risk packs:**
- Stop: `(unset)` (method: swing, padding: 0.03, lookback: 5)
- Target: `(unset)` (signal exit only)
- Time exit: `(unset)` (method: --, max_bars: --)

**Confluences:** 1d-MACD_LINE-M>S-, 5m-UTBOT-BEAR

**Data lookback:** 90 days, mode: `days`

**Recreation checklist:**
- [ ] New strategy created and named `Recreated 66 — AAPL LONG - Mass #23`
- [ ] Verified entry trigger picker found the same/equivalent
- [ ] Saved + waited ~30 min for first signal
- [ ] First alert fired (check alert count on Strategies list page)
- [ ] Deleted strategy 66


## Strategy 111 — SPY LONG - 1

- **Symbol:** SPY
- **Direction:** LONG
- **Timeframe:** 10Sec
- **Session:** Extended Hours
- **Created:** 2026-04-15T20:14:30.155384+00:00
- **Original FW Start:** 2026-04-15T20:14:30.155394+00:00

**Triggers:**
- Entry: `swing_123_test_default_bullish_c2_detected`
- Exit: (none — signal-only? check original)

**Risk packs:**
- Stop: `swing_2r` (method: swing, padding: 0.05, lookback: 5)
- Target: `` (signal exit only)
- Time exit: `max_hold_4` (method: max_hold_bars, max_bars: 4)

**Confluences:** 1h-SWING_123_TEST-NEUTRAL

**Data lookback:** 2 days, mode: `Days`

**Recreation checklist:**
- [ ] New strategy created and named `Recreated 111 — SPY LONG - 1`
- [ ] Verified entry trigger picker found the same/equivalent
- [ ] Saved + waited ~30 min for first signal
- [ ] First alert fired (check alert count on Strategies list page)
- [ ] Deleted strategy 111


## Strategy 122 — SPY LONG 1Min Mass #136

- **Symbol:** SPY
- **Direction:** LONG
- **Timeframe:** 1Min
- **Session:** RTH
- **Created:** 2026-04-22T03:09:42.866084+00:00
- **Original FW Start:** 2026-04-22T03:09:42.866098+00:00

**Triggers:**
- Entry: `swing_123_default_bull_c3`
- Exit: `swing_123_default_bear_c2`

**Risk packs:**
- Stop: `(unset)` (method: swing, padding: 0.05, lookback: 5)
- Target: `(unset)` (signal exit only)
- Time exit: `(unset)` (method: --, max_bars: --)

**Confluences:** 15m-MACD_HISTOGRAM-H+dn

**Data lookback:** 180 days, mode: `(unset)`

**Recreation checklist:**
- [ ] New strategy created and named `Recreated 122 — SPY LONG 1Min Mass #136`
- [ ] Verified entry trigger picker found the same/equivalent
- [ ] Saved + waited ~30 min for first signal
- [ ] First alert fired (check alert count on Strategies list page)
- [ ] Deleted strategy 122


## Strategy 129 — extend test new table

- **Symbol:** SPY
- **Direction:** LONG
- **Timeframe:** 10Sec
- **Session:** Extended Hours
- **Created:** 2026-04-24T21:19:06.680341+00:00
- **Original FW Start:** 2026-04-24T21:19:06.680355+00:00

**Triggers:**
- Entry: `swing_123_test_default_bullish_c2_detected`
- Exit: `swing_123_default_bear_c2`

**Risk packs:**
- Stop: `swing_2r` (method: swing, padding: 0.05, lookback: 5)
- Target: `` (signal exit only)
- Time exit: `max_hold_10` (method: max_hold_bars, max_bars: 10)

**Confluences:** none

**Data lookback:** 2 days, mode: `Days`

**Recreation checklist:**
- [ ] New strategy created and named `Recreated 129 — extend test new table`
- [ ] Verified entry trigger picker found the same/equivalent
- [ ] Saved + waited ~30 min for first signal
- [ ] First alert fired (check alert count on Strategies list page)
- [ ] Deleted strategy 129
