# Phase 31: Polygon.io Data Provider Migration — Implementation Spec

**Status:** Approved — Building on `dev` branch
**Date:** 2026-03-20
**Backup branch:** `main-backup-pre-31`
**Plan file:** `/home/kevin/.claude/plans/sequential-fluttering-sutherland.md`

---

## 1. Overview

Replace Alpaca market data (REST + WebSocket) with Polygon.io/Massive. Alpaca stays as broker. Build order: 31A → 31B → 31C → 31D → 31E.

---

## 2. Phase 31A: Polygon REST Integration

### Self-Test Checkpoints

- [ ] **A1:** `_to_polygon_ticker("SPY")` == `"SPY"`, `_to_polygon_ticker("BTC/USD")` == `"X:BTCUSD"`
- [ ] **A2:** `_polygon_timespan("1Min")` == `(1, "minute")`, `_polygon_timespan("5Sec")` == `(5, "second")`
- [ ] **A3:** `load_from_polygon("SPY", days=5, timeframe="1Min")` returns DataFrame with ~1,950 bars, columns: open/high/low/close/volume/vwap/trade_count
- [ ] **A4:** `load_from_polygon_crypto("BTC/USD", days=2, timeframe="1Min")` returns ~2,880 bars
- [ ] **A5:** Parity: Alpaca vs Polygon for SPY 1Min on same date — row counts within 2, close prices within $0.02
- [ ] **A6:** Pagination: SPY 1Min for 60 days loads >23K bars without error
- [ ] **A7:** Session filter: `session="RTH"` returns no bars outside 9:30-16:00 ET
- [ ] **A8:** Mock fallback: `POLYGON_API_KEY=""` → mock data returned
- [ ] **A9:** Sub-minute: `load_from_polygon("SPY", days=1, timeframe="30Sec")` returns bars
- [ ] **A10:** `DATA_PROVIDER="polygon"` routes through Polygon; `"alpaca"` routes through Alpaca
- [ ] **A11:** All 27 unified engine parity tests pass

---

## 3. Phase 31B: Polygon WebSocket Integration

### Self-Test Checkpoints

- [ ] **B1:** WebSocket auth succeeds (status: auth_success)
- [ ] **B2:** AM.SPY subscription produces completed minute bars
- [ ] **B3:** Bar routes: SymbolHub → monitor.on_bar_close() → signals generated
- [ ] **B4:** Gap fill after 3+ minute disconnect
- [ ] **B5:** Dual streams: stocks (AM.SPY) + crypto (XA.X:BTCUSD) both produce bars
- [ ] **B6:** Warmup + live continuity: no indicator jumps at transition
- [ ] **B7:** engine_status.json reports connected: true after first bar
- [ ] **B8:** WS bars match REST bars for same 30-minute window
- [ ] **B9:** Stop/restart preserves position state

---

## 4. Phase 31C: Per-Second Bars for L-Type Detection

### Self-Test Checkpoints

- [ ] **C1:** A.SPY delivers per-second bars during market hours
- [ ] **C2:** L-type entry fires when 1s bar high/low crosses trigger level
- [ ] **C3:** No false triggers over 1-hour test period
- [ ] **C4:** Per-second bars only subscribed for symbols with L-type strategies

---

## 5. Phase 31D: Remove Alpaca Data Dependencies

### Self-Test Checkpoints

- [ ] **D1:** No Alpaca data imports remain (grep confirms)
- [ ] **D2:** App starts cleanly with only POLYGON_API_KEY
- [ ] **D3:** Strategy Builder backtest works end-to-end
- [ ] **D4:** Ralph engine connects and generates alerts via Polygon WS
- [ ] **D5:** Railway deploy works with POLYGON_API_KEY env var
- [ ] **D6:** All 27 parity tests pass
- [ ] **D7:** UI shows "Polygon" instead of "Alpaca" in status displays

---

## 6. Phase 31E: Sub-Minute Backtesting

### Self-Test Checkpoints

- [ ] **E1:** 30Sec appears in Strategy Builder timeframe dropdown
- [ ] **E2:** 30Sec backtest loads data and generates trades
- [ ] **E3:** 5Sec bars resample correctly to 1Min
- [ ] **E4:** MTF confluence with sub-minute secondary TF works
- [ ] **E5:** Performance: 30Sec 5-day backtest completes in <5s

---

## 7. End-to-End Verification

After all phases:
1. [ ] Create a new strategy using Polygon data — backtest, save, forward test
2. [ ] Live monitoring via Ralph on Polygon WS generates alerts correctly
3. [ ] Portfolio Live Dashboard shows trades from Polygon-sourced alerts
4. [ ] Sub-minute backtest produces meaningful results
5. [ ] No Alpaca data API calls remain (only broker references kept)
6. [ ] Railway cloud deployment fully functional
