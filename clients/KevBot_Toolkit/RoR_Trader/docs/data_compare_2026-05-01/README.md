# Four-way data comparison — 2026-05-01

Comparing how the same OHLCV bar looks across four sources to characterize
WS-vs-REST-vs-TV divergence.

## Sources being compared

| Source | Origin | Captured by |
|---|---|---|
| `first_*` (cache) | Polygon WS at first write — what the engine SAW at decision time | `live_bars` table, `first_close` etc. columns |
| `close` (cache) | Polygon WS after rebroadcast corrections within 15-min FINRA window | `live_bars` table, latest `close` etc. columns |
| Polygon REST | Polygon's settled aggregate after end-of-day reconciliation | `data_loader.load_market_data()` |
| TradingView | Independent vendor (TV builds bars from CQS/UTP feeds) | TV "Export chart data" CSV → paste here |

## What to grab from TradingView

For each CSV file in this directory, open SPY/AAPL/TSLA/TSLL on the
matching timeframe in TradingView, then **File → Export chart data**
(or right-click chart → Export chart data). Paste the CSV contents
into the matching file.

**Time window**: today (2026-05-01), 09:30 ET (= 13:30 UTC) through
whenever you do the export. We'll automatically trim to the overlap
with our cache when running the comparison.

**Don't worry about TV's CSV format** — I'll auto-detect headers when
parsing. Just paste whatever TV gives you.

## Files

Priority 1 — drift-vs-timeframe story for one symbol (SPY):
- `tv_SPY_10Sec.csv` — sub-minute, knife-edge sensitive
- `tv_SPY_1Min.csv` — workhorse
- `tv_SPY_15Min.csv` — coarse, drift should largely wash out

Priority 2 — liquidity comparison (different symbols):
- `tv_AAPL_1Min.csv` — different liquidity from SPY
- `tv_AAPL_5Min.csv` — mid TF, different symbol
- `tv_TSLA_10Sec.csv` — sub-minute on a higher-volatility name

Priority 3 — TF coverage extender:
- `tv_TSLL_10Min.csv` — only TF we have in cache for that range

## Notes

- TradingView export typically gives 1Min and coarser by default.
  10Sec / 30Sec exports may require higher-tier plan; if not
  supported, leave that CSV empty and we'll do 3-way for those
  combos (cache `first` vs cache `close` vs REST).
- TV defaults to RTH-only on most chart settings; that's fine for
  comparison since our cache also runs RTH session for these
  symbols.
- If you grab files I didn't list, just drop them in this dir
  with the same `tv_<SYMBOL>_<TF>.csv` naming and I'll pick them
  up automatically.
