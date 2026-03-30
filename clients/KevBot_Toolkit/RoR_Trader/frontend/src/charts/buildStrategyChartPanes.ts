'use client';

/**
 * buildStrategyChartPanes — shared chart construction logic for
 * Strategy Detail and Strategy Builder pages.
 *
 * Takes bars, trades, indicators, and preferences, returns PaneConfig[]
 * ready to pass to SyncedChartPane. This ensures both pages render
 * identical charts.
 */

import type { PaneConfig, SeriesConfig } from '@/charts/SyncedChartPane';

const INDICATOR_COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#00BCD4', '#9C27B0', '#FFC107', '#795548'];

interface ChartBuildOptions {
  bars: any[];
  trades: any[];
  direction: string;
  overlayNames?: string[];
  oscNames?: string[];
  heatmapConds?: any[];
  showConditions?: boolean;
  showTriggers?: boolean;
  chartPrefs?: {
    entryColor: string;
    exitWinColor: string;
    exitLossColor: string;
    exitStopColor: string;
    exitBarCountColor: string;
    exitHybridColor?: string;
    showLabels: boolean;
  };
}

const DEFAULT_PREFS = {
  entryColor: '#4CAF50',
  exitWinColor: '#4CAF50',
  exitLossColor: '#F44336',
  exitStopColor: '#FF9800',
  exitBarCountColor: '#26A69A',
  exitHybridColor: '#FF9800',
  showLabels: true,
};

export function buildStrategyChartPanes(opts: ChartBuildOptions): PaneConfig[] {
  const {
    bars, trades, direction,
    overlayNames = [], oscNames = [], heatmapConds = [],
    showConditions = true, showTriggers = true,
    chartPrefs = DEFAULT_PREFS,
  } = opts;

  if (bars.length === 0) return [];

  const prefs = { ...DEFAULT_PREFS, ...chartPrefs };
  const firstBarTime = new Date(bars[0].timestamp).getTime();
  const lastBarTime = new Date(bars[bars.length - 1].timestamp).getTime();

  // ---- Trade markers (arrows) ----
  const tradeMarkers = !showTriggers ? [] : trades.flatMap((t: any) => {
    const m: any[] = [];
    const dir = direction;
    const entryTime = t.entry_time || t.entryTime;
    const exitTime = t.exit_time || t.exitTime;
    const entryMs = entryTime && entryTime !== '--' ? new Date(entryTime).getTime() : 0;
    const exitMs = exitTime && exitTime !== '--' ? new Date(exitTime).getTime() : 0;
    const rMult = t.r_multiple ?? t.rMultiple ?? t.pnlR ?? 0;
    const exitReason = t.exit_reason || t.exitReason || '';

    if (entryMs >= firstBarTime && entryMs <= lastBarTime) {
      m.push({
        time: entryTime, position: dir === 'LONG' ? 'belowBar' : 'aboveBar',
        shape: dir === 'LONG' ? 'arrowUp' : 'arrowDown',
        color: prefs.entryColor, text: prefs.showLabels ? 'Entry' : '', size: 1,
      });
    }
    if (exitMs >= firstBarTime && exitMs <= lastBarTime) {
      let color = rMult >= 0 ? prefs.exitWinColor : prefs.exitLossColor;
      if (exitReason === 'stop_loss') color = prefs.exitStopColor;
      else if (exitReason === 'bar_count_exit') color = prefs.exitBarCountColor;
      else if (exitReason === 'opposite_signal' || exitReason === 'time_exit') color = prefs.exitHybridColor || '#FF9800';
      m.push({
        time: exitTime, position: dir === 'LONG' ? 'aboveBar' : 'belowBar',
        shape: 'arrowDown', color,
        text: prefs.showLabels ? `${rMult >= 0 ? '+' : ''}${rMult.toFixed(1)}R` : '', size: 1,
      });
    }
    return m;
  });

  // ---- Price-level cross markers (+) ----
  const priceCrossEntries: any[] = [];
  const priceCrossMarkers: any[] = [];
  const seenEntry = new Set<string>();
  if (showTriggers) {
    const barTimestamps = bars.map((b: any) => b.timestamp);
    const snapToBar = (tradeTime: string): string | null => {
      const tradeMs = new Date(tradeTime).getTime();
      let bestTs = barTimestamps[0];
      let bestDist = Infinity;
      for (const ts of barTimestamps) {
        const dist = Math.abs(new Date(ts).getTime() - tradeMs);
        if (dist < bestDist) { bestDist = dist; bestTs = ts; }
      }
      return bestDist < 120000 ? bestTs : null;
    };

    for (const t of trades) {
      const entryTime = t.entry_time || t.entryTime;
      const entryPrice = t.entry_price ?? t.entryPrice ?? 0;
      if (entryTime && entryTime !== '--' && entryPrice > 0) {
        const snapped = snapToBar(entryTime);
        if (snapped && !seenEntry.has(snapped)) {
          seenEntry.add(snapped);
          priceCrossEntries.push({ time: snapped, value: entryPrice });
          priceCrossMarkers.push({ time: snapped, position: 'inBar', shape: 'cross', color: prefs.entryColor, text: '', size: 1 });
        }
      }
    }
  }

  // ---- Build panes ----
  const chartPanes: PaneConfig[] = [];

  // Pane 1: Confluence heatmap
  if (showConditions && heatmapConds.length > 0) {
    const n = heatmapConds.length;
    const hmSeries: SeriesConfig[] = heatmapConds.map((cond: any, idx: number) => ({
      type: 'Histogram' as const,
      data: bars.map((b: any) => {
        const stateVal = b[`_state_${cond.column}`];
        const isMet = stateVal != null && stateVal === cond.needed_state;
        return { time: b.timestamp, value: n - idx, color: isMet ? 'rgba(76,175,80,0.8)' : 'rgba(244,67,54,0.4)' };
      }),
      options: { priceLineVisible: false, lastValueVisible: false, title: cond.label },
    }));
    chartPanes.push({ id: 'heatmap', height: Math.max(50, n * 20 + 10), series: hmSeries, hideTimeAxis: true });
  }

  // Pane 2: Price chart + overlays + trade markers
  const priceSeries: SeriesConfig[] = [
    {
      type: 'Candlestick',
      data: bars.map((b: any) => ({ time: b.timestamp, open: b.open, high: b.high, low: b.low, close: b.close })),
      markers: tradeMarkers,
    },
  ];

  // Cross markers
  if (priceCrossEntries.length > 0) {
    priceSeries.push({
      type: 'Line',
      data: priceCrossEntries,
      options: { color: prefs.entryColor, lineVisible: false, pointMarkersVisible: false, priceLineVisible: false, crosshairMarkerVisible: false, lastValueVisible: false, title: '' },
      markers: priceCrossMarkers,
    });
  }

  // Overlay indicators
  for (let i = 0; i < overlayNames.length; i++) {
    const col = overlayNames[i];
    priceSeries.push({
      type: 'Line',
      data: bars.filter((b: any) => b[col] != null).map((b: any) => ({ time: b.timestamp, value: b[col] })),
      options: { color: INDICATOR_COLORS[i % INDICATOR_COLORS.length], lineWidth: 2, title: col.replace(/_/g, ' ') },
    });
  }
  chartPanes.push({ id: 'price', height: 350, series: priceSeries });

  // Pane 3: Oscillators
  if (oscNames.length > 0) {
    const oscSeries: SeriesConfig[] = [];
    for (let i = 0; i < oscNames.length; i++) {
      const col = oscNames[i];
      const isHist = col.toLowerCase().includes('hist');
      oscSeries.push({
        type: isHist ? 'Histogram' : 'Line',
        data: bars.filter((b: any) => b[col] != null).map((b: any) => ({
          time: b.timestamp, value: b[col],
          ...(isHist ? { color: b[col] >= 0 ? 'rgba(76,175,80,0.6)' : 'rgba(244,67,54,0.6)' } : {}),
        })),
        options: isHist
          ? { priceLineVisible: false, lastValueVisible: false, title: col.replace(/_/g, ' ') }
          : { color: INDICATOR_COLORS[(i + 4) % INDICATOR_COLORS.length], lineWidth: 1.5, priceLineVisible: false, title: col.replace(/_/g, ' ') },
      });
    }
    chartPanes.push({ id: 'oscillator', height: 120, series: oscSeries });
  }

  return chartPanes;
}
