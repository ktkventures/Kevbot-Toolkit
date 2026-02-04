# Alpaca API Exploration

**Date:** February 3, 2026
**Purpose:** Understand Alpaca's data structure for RoR Trader integration

---

## API Overview

**Base URL:** `https://data.alpaca.markets/v2/stocks/bars`
**Authentication:** API Key + Secret Key in headers
**SDK:** `alpaca-py` (official Python SDK)

---

## Pricing Tiers

| Plan | Price | Rate Limit | Historical Data | Real-Time |
|------|-------|------------|-----------------|-----------|
| **Basic (Free)** | $0/mo | 200 calls/min | Since 2016 (15-min delayed) | IEX only |
| **Algo Trader Plus** | $99/mo | 10,000 calls/min | Since 2016 (no restriction) | All exchanges |

**Recommendation for MVP:** Start with Basic (free) tier for development. Historical data is available for backtesting. Real-time data from IEX is sufficient for forward testing initially.

---

## Data Structure

### Bar (OHLCV) Response

```json
{
  "bars": {
    "SPY": [
      {
        "t": "2026-02-03T14:30:00Z",    // timestamp (RFC-3339)
        "o": 498.12,                     // open
        "h": 498.55,                     // high
        "l": 497.89,                     // low
        "c": 498.32,                     // close
        "v": 125432,                     // volume
        "n": 1523,                       // trade count
        "vw": 498.21                     // volume-weighted avg price
      }
    ]
  },
  "next_page_token": "abc123..."         // for pagination
}
```

### Available Timeframes

| Category | Options |
|----------|---------|
| Minutes | `1Min`, `5Min`, `15Min`, `30Min` (up to `59Min`) |
| Hours | `1Hour`, `4Hour` (up to `23Hour`) |
| Daily | `1Day` |
| Weekly | `1Week` |
| Monthly | `1Month`, `3Month`, `6Month`, `12Month` |

---

## Python SDK Usage

### Installation
```bash
pip install alpaca-py
```

### Getting Historical Bars

```python
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime

# Initialize client (requires API keys for stock data)
client = StockHistoricalDataClient(
    api_key="YOUR_API_KEY",
    secret_key="YOUR_SECRET_KEY"
)

# Request parameters
request = StockBarsRequest(
    symbol_or_symbols=["SPY", "AAPL"],
    timeframe=TimeFrame.Minute,          # 1-minute bars
    start=datetime(2026, 1, 1),
    end=datetime(2026, 2, 1)
)

# Fetch data
bars = client.get_stock_bars(request)

# Convert to pandas DataFrame (multi-index: symbol, timestamp)
df = bars.df

# Access single symbol
spy_bars = df.loc["SPY"]
```

### TimeFrame Options in SDK

```python
from alpaca.data.timeframe import TimeFrame

TimeFrame.Minute      # 1Min
TimeFrame.Hour        # 1Hour
TimeFrame.Day         # 1Day
TimeFrame.Week        # 1Week
TimeFrame.Month       # 1Month

# Custom timeframes
TimeFrame(5, TimeFrameUnit.Minute)   # 5Min
TimeFrame(4, TimeFrameUnit.Hour)     # 4Hour
```

---

## Data We Need for RoR Trader

### For Backtesting
- **Historical OHLCV bars** at 1-minute resolution
- Multiple timeframes calculated from 1-minute data (or fetched directly)
- Volume data for RVOL calculations
- VWAP (provided in API response as `vw`)

### For Forward Testing
- Real-time or near-real-time bar updates
- WebSocket streaming (available in Alpaca)

### For Indicators/Interpreters
We'll calculate these from the raw bar data:
- EMAs (8, 21, 50)
- MACD (12, 26, 9)
- VWAP (already in response)
- Relative Volume (compare current to historical average)
- Swing highs/lows (from price action)

---

## Next Steps

1. [x] Document API structure
2. [ ] Set up Alpaca account and get API keys
3. [ ] Create test script to fetch sample data
4. [ ] Verify data quality and completeness
5. [ ] Design data storage schema
6. [ ] Build indicator calculation layer

---

## Sources

- [Alpaca Historical Bars API](https://docs.alpaca.markets/reference/stockbars)
- [Getting Started with Market Data](https://docs.alpaca.markets/docs/getting-started-with-alpaca-market-data)
- [alpaca-py GitHub](https://github.com/alpacahq/alpaca-py)
- [Python SDK Market Data Guide](https://alpaca.markets/sdks/python/market_data.html)
