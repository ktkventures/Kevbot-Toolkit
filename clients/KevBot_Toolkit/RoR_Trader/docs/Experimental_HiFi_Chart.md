# Experimental: HiFi Sub-Second Charting System

**Status:** CONCEPT — Documenting for future consideration
**Date:** 2026-03-22
**Depends on:** Phase 31F (HiFi Backtest & Fidelity System), Polygon 1-second bars

---

## Concept

A high-fidelity charting mode where the x-axis is based on **seconds** rather than candles. Candles are overlaid on the time-based axis at the user's chosen aggregation (1min, 5min, etc.), while all markers, confluence states, and indicators are plotted at their exact sub-second timestamps.

Inspired by **BookMap** — the time-based x-axis approach applied to the RoR Trader confluence/fidelity system.

## Why This Fits RoR Trader

The 31F HiFi system already computes sub-candle data:
- Exact second of level crosses (`[L]` type entries)
- Exact second stops/targets are hit within a bar
- `hifi_exit_second` timestamp per resolved trade
- Polygon 1-second bar data already being fetched for Pass 2

The charting would visualize what the engine already knows.

## What It Would Show

### Second-by-Second Price
- X-axis: wall-clock time in seconds
- Price plotted as a continuous line or 1-second micro-candles
- Overlaid aggregated candles (1min, 5min, etc.) shown as semi-transparent boxes on top

### Precise Marker Placement
- `+` (backtest entry/exit) plotted at the exact second the engine determined the fill
- `x` (alert entry/exit) plotted at the exact second the alert fired
- Visible horizontal offset between `+` and `x` = measurable execution latency/slippage
- Much more precise than current candle-level marker snapping

### Confluence Heatmap at Second Resolution
- Confluence conditions (EMA_STACK, MACD_LINE, etc.) hold their state between candle closes
- On the second a new candle closes and re-evaluates, the heatmap color transitions
- Cross-TF confluence changes visible at the exact second the higher TF bar closes
- Yellow "speculative" state visible on forming bars

### Indicators
- Indicators sync to candle boundaries (EMA updates every 1min, not every second)
- Displayed as stepped lines that update at candle close times
- Price moves freely underneath while indicators hold steady between updates
- Demonstrates the relationship between forming price and confirmed indicator values

## Use Cases

### 1. Live Chart (Primary)
- Real-time price with second-level precision
- See alerts fire in real-time with exact timestamp
- Watch confluence states transition as candles close
- Monitor execution latency between signal and fill

### 2. Trade Inspection (Drill-Down)
- Zoom into a specific trade's entry/exit bars
- See second-by-second what happened during the bar
- Verify HiFi resolution: did stop or target get hit first?
- Compare alert price vs backtest price at second level

### 3. Alert Analysis
- See exact timing: when did the bar close → when did the engine detect → when did the alert fire → when did the webhook send
- Measure the full latency pipeline
- Identify if any systematic delays exist

## Technical Feasibility

### Data Source
- **Polygon 1-second bars** — already available on the $199/mo Stocks Advanced plan
- REST endpoint: `GET /v2/aggs/ticker/{ticker}/range/1/second/{from}/{to}`
- WebSocket: real-time 1-second aggregates via `AM.ticker` subscription
- Rate limits: 100 calls/second (sufficient for single-ticker drill-down)

### Data Volume
| View | Data Points | Fetch Time |
|------|-------------|------------|
| 5 minutes | 300 | Instant |
| 1 hour | 3,600 | <1s |
| 1 day (RTH) | 23,400 | ~2s |
| 1 week | 117,000 | ~5s |
| 1 month | ~500,000 | Not recommended for full fetch |

### Rendering
- Canvas-based renderers (Lightweight Charts, ECharts) handle 100K+ points easily
- Level-of-detail (LOD) aggregation needed for smooth zoom:
  - Zoomed out: 1-minute candles (standard view)
  - Zoomed in: 1-second micro-candles or line
  - Intermediate: 5-second, 15-second aggregations

### Zoom Behavior
- Mousewheel zoom transitions between aggregation levels
- Zoomed all the way out: looks like a normal candle chart
- Zoomed all the way in: individual seconds visible, markers precisely placed
- Crosshair shows exact timestamp (HH:MM:SS) instead of candle time

### Challenges
1. **Zoom LOD transitions** — smoothly transitioning between 1-second and 1-minute aggregation is complex. BookMap solves this with pre-computed LOD levels.
2. **Indicator rendering at mixed resolutions** — price at 1-second, EMA at 1-minute. Need visual distinction (stepped lines, different opacity).
3. **Historical data limits** — 1-second data from Polygon is available for limited history. Not suitable for multi-month backtests at this resolution.
4. **Memory management** — need to window the data (only load what's visible + buffer) rather than loading entire days.

## Chart Library Options

| Library | Sub-Second Support | LOD Zoom | Performance | Notes |
|---------|-------------------|----------|-------------|-------|
| **Lightweight Charts** | Partial — custom time scale possible | Manual implementation | Excellent | TradingView foundation, would need custom series for LOD |
| **ECharts** | Yes — custom axis formatting | Built-in dataZoom | Good | Richer tooltip support, hover shows OHLCV per second |
| **D3.js** | Full control | Manual implementation | Depends | Maximum flexibility but most work |
| **Custom Canvas** | Full control | Full control | Maximum | Most work, most control over zoom/LOD |
| **Scichart** | Yes — built for financial | Built-in | Excellent | Commercial license, purpose-built for this |

## Recommended Approach

### Phase 1: Proof of Concept
- Build a standalone HiFi chart component
- Fetch 1-hour of 1-second data from Polygon REST
- Render with ECharts (best hover/tooltip support)
- Plot a few mock markers at second-level timestamps
- Test zoom performance

### Phase 2: Integration
- Add "HiFi View" button to existing price charts
- Opens the HiFi chart in a modal or replaces the chart temporarily
- Auto-fetches 1-second data for the visible time range
- Overlays confluence heatmap and markers

### Phase 3: Live Mode
- Connect to Polygon WebSocket for real-time 1-second data
- Append new data points to the chart in real-time
- Show alerts firing in real-time at second precision
- Auto-scroll with price

## Where This Lives in the App

NOT the default chart everywhere. Available as:
- **"HiFi" toggle button** on any price chart (strategy detail, portfolio detail, alerts)
- **Default for live chart** view (when monitoring active strategies)
- **Trade inspection modal** — click a trade to see the second-by-second view of its entry/exit
- **Setting in Display Preferences** — "Default to HiFi chart for live views" toggle

## Relationship to Existing Settings

The Display Preferences page already has:
- Chart library selector (Lightweight vs ECharts)
- Candle style, colors, grid lines
- Position marker shapes and colors

HiFi chart would inherit these settings where applicable:
- Same marker shapes/colors for `+` and `x`
- Same candle color theme for the overlay candles
- Same grid line preference
- Additional settings: LOD transition behavior, default zoom level, micro-candle vs line mode

---

## Decision Needed

- [ ] Is this worth pursuing as a Phase 39 or later feature?
- [ ] Should it be the default live chart or opt-in only?
- [ ] Which chart library to use (ECharts recommended for hover tooltips)?
- [ ] Scope: live-only or also available for historical trade inspection?
