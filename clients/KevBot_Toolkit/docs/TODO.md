# KevBot Toolkit - TODO / Session Notes

## Next Session Priority

### Continue TradingView Toolkit Development

Return to building the TradingView indicator side modules and features. The Trade Analysis Tool POC is complete and ready to wire up once the toolkit's export structure is finalized.

**Pending Toolkit Work:**
- Build UT Bot library (preferred trigger module)
- Build remaining side modules: VWAP, RVOL, SR Channel, Swing 123
- Manually verify MACD Divergence behavior against chart interpretation
- Complete backtest KPI calculations in toolkit
- Add Side Modules 3-10 (follow existing pattern)

---

## Completed Items

### ✅ Trade Analysis Tool (Python/Streamlit) - POC Complete

**Location:** `tools/trade_analyzer/`

**What was built:**
- Streamlit web app for analyzing confluence combinations
- Mock data generator mimicking TradingView CSV export format
- Two analysis modes:
  - **Drill-Down**: Start with best single factor, iteratively add confluences
  - **Auto-Search**: Combinatorial search for best N-factor combinations
- "Confluence Record" concept: atomic units combining Timeframe + Evaluator + State (e.g., "1M-EMA-SML")
- KPI Dashboard: Trades, Win Rate, Profit Factor, Daily P&L (R and $)
- P&L Settings: Fixed vs Compounding risk modes, starting balance, risk per trade
- Interactive equity curve with high water mark
- Export TradingView Parameters button (placeholder for future integration)

**Tech Stack:** Python, Streamlit, Pandas, Plotly (~800 lines)

**Status:** POC complete. Ready to integrate with real toolkit export data once format is finalized.

---

## Other Pending Items

- [ ] Manually verify MACD Divergence behavior against chart interpretation
- [ ] Build UT Bot library (preferred trigger module)
- [ ] Build remaining side modules: VWAP, RVOL, SR Channel, Swing 123

---

*Last Updated: January 30, 2026*
