# KevBot Trade Analyzer

Proof of concept tool for analyzing exported trade data and finding optimal confluence combinations.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Core Concept: Confluence Records

A **confluence record** is the atomic unit of analysis, combining:
- **Timeframe** (e.g., "1M", "5M", "1H")
- **Evaluator** (e.g., "EMA", "MACD") - the module analyzing the data
- **State** (e.g., "SML", "M>S↑") - the condition output

Example: `1M-EMA-SML` means "On the 1-minute chart, the EMA Stack evaluator shows a bullish stack (Short > Medium > Long)"

## Features

### Analysis Modes

**Drill-Down Mode:**
- Start with the best-performing single confluence record
- Click to add it as a filter
- See what additional factors pair well with your selection
- Iteratively build your ideal confluence combination

**Auto-Search Mode:**
- Automatically search for best N-factor combinations
- Filter by minimum trades, win rate, profit factor
- Click "Apply" to use any discovered combination

### KPI Dashboard
- **Trades**: Number of trades matching current filters
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Gross profit / Gross loss
- **Daily P&L**: Total R / trading days (shown in R and $)
- **Total P&L**: Cumulative R-multiple (shown in R and $)
- **Final Balance**: Account value after all trades

### P&L Settings
Configure your financial model:
- **Risk Mode**: Fixed (same $ per trade) or Compounding (% of balance)
- **Starting Balance**: Initial account size
- **Risk per Trade**: Dollar amount or percentage

### Visualizations
- Interactive equity curve with high water mark
- R-multiple distribution histogram
- Trade list with confluence details

## Status

This is a proof of concept. The mock data format mirrors what TradingView exports will look like.

### What's included:
- Mock data with 2 evaluators (EMA Stack, MACD Simple)
- 6 configurable timeframes
- Entry/exit signal detection
- Confluence combination analysis (AND logic)
- Financial modeling (fixed/compounding)

### Future enhancements (after toolkit integration):
- Real CSV import from TradingView toolkit exports
- Export TradingView Parameters (auto-configure indicator settings)
- Support for all 6 side module slots
- Additional evaluator libraries
- SL/TP tracking and analysis
