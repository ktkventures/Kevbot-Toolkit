# KevBot Toolkit - TODO / Session Notes

## Next Session Priority

### Build Trade Analysis Tool (Python/Streamlit)

**Goal:** Create a web-based analysis tool to find optimal confluence combinations from exported trade data.

**Requirements:**
- Import CSV data exports from TradingView (the toolkit's data export feature)
- Input fields for entry trigger and exit trigger selection
- Calculate KPIs (win rate, R-multiple, profit factor, etc.) grouped by confluence combinations
- Rank and display "best performing confluence setups"
- Interactive filtering: e.g., "Show trades where Trigger A fired AND EMA Stack was SML on TF1-3"
- Heatmaps or visualizations showing which confluences correlate with positive outcomes

**Tech Stack:**
- Python
- Streamlit (simple web UI)
- Pandas for data processing
- ~200-300 lines estimated

**Why web app over spreadsheet:**
- Multi-dimensional analysis (triggers × confluences × TFs × outcomes)
- Finding "best combinations" is statistical analysis that's hard in spreadsheet formulas
- Code can be written directly vs. giving spreadsheet instructions
- Better interactivity and visualizations

---

## Other Pending Items

- [ ] Manually verify MACD Divergence behavior against chart interpretation
- [ ] Build UT Bot library (preferred trigger module)
- [ ] Build remaining side modules: VWAP, RVOL, SR Channel, Swing 123

---

*Last Updated: January 29, 2026*
