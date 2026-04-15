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
  /** Live alerts — rendered as xcross price-level markers so users can
   * distinguish alert fills from backtest/forward trade fills. */
  alerts?: any[];
  direction: string;
  overlayNames?: string[];
  oscNames?: string[];
  heatmapConds?: any[];
  showConditions?: boolean;
  showTriggers?: boolean;
  /** Timeframe duration in ms — used to shift C-type timestamps to next bar open */
  tfMs?: number;
  /** Column name from pack's plot_config.candle_color_column — overrides candle colors */
  candleColorColumn?: string;
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

/** Check if a column's values are plottable (numeric, not boolean/string).
 * Scans all bars (not just the first 10) because indicators with long warmup
 * periods (e.g., Stochastic, EMA 200) have NaN values for the first N bars. */
function isPlottableColumn(bars: any[], colName: string): boolean {
  for (const bar of bars) {
    const val = bar[colName];
    if (val === null || val === undefined) continue;
    if (typeof val === 'boolean') return false;
    if (typeof val === 'string') return false;
    if (typeof val === 'number' && !isNaN(val)) return true;
  }
  return false; // all null/undefined = skip
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
    bars, trades, alerts = [], direction,
    overlayNames = [], oscNames = [], heatmapConds = [],
    showConditions = true, showTriggers = true,
    tfMs = 60000,
    candleColorColumn,
    chartPrefs = DEFAULT_PREFS,
  } = opts;

  // Filter overlay/oscillator names to only plottable numeric columns
  const filteredOverlays = overlayNames.filter((col) => isPlottableColumn(bars, col));
  const filteredOscs = oscNames.filter((col) => isPlottableColumn(bars, col));

  // L-type exit reasons fire mid-bar (intra-bar touch). Other exec types
  // (C, CC) fire at bar close — shift to next bar open for realistic plotting.
  const ALWAYS_L_TYPE_EXITS = new Set(['unconfirmed_hl']);

  // Determine exec type from trigger suffix for C-type shift
  const getExecType = (id: string) => {
    if (!id) return 'C';
    if (id.endsWith('_lc') || id.endsWith('_hm') || id.endsWith('_hl')) return 'L';
    if (id.endsWith('_ib')) return 'L';
    if (id.endsWith('_cc')) return 'CC';
    return 'C';
  };

  // Resolve the effective exec type for a trade exit, taking the per-trade
  // stop/target exec_type into account (M5.5 — stops and targets now support
  // L/C/LC/CC dispatch). Falls back to L for unconfirmed_hl, C otherwise.
  const getExitExecType = (trade: any): string => {
    const reason = trade.exit_reason || trade.exitReason || '';
    if (ALWAYS_L_TYPE_EXITS.has(reason)) return 'L';
    if (reason === 'stop_loss' || reason === 'stop') {
      return trade.stop_exec_type || trade.stopExecType || 'L';
    }
    if (reason === 'target') {
      return trade.target_exec_type || trade.targetExecType || 'L';
    }
    return 'C';
  };

  // Shift C-type timestamps to next bar open for realistic plotting
  const shiftIfCType = (iso: string, execType: string) => {
    if (!iso || iso === '--') return iso;
    if (execType === 'L' || execType === 'LC') return iso; // L-type: no shift
    try {
      const d = new Date(iso);
      if (isNaN(d.getTime())) return iso;
      return new Date(d.getTime() + tfMs).toISOString();
    } catch { return iso; }
  };

  if (bars.length === 0) return [];

  const prefs = { ...DEFAULT_PREFS, ...chartPrefs };
  const _safeMs = (v: string) => {
    const d = new Date(v);
    if (isNaN(d.getTime())) { console.warn('[ChartPanes] Invalid date:', v); return 0; }
    return d.getTime();
  };
  const firstBarTime = bars.length > 0 ? _safeMs(bars[0].timestamp) : 0;
  const lastBarTime = bars.length > 0 ? _safeMs(bars[bars.length - 1].timestamp) : Infinity;

  // ---- Trade markers (arrows) with C-type shift ----
  const tradeMarkers = !showTriggers ? [] : trades.flatMap((t: any) => {
    const m: any[] = [];
    const dir = direction;
    const rawEntryTime = t.entry_time || t.entryTime;
    const rawExitTime = t.exit_time || t.exitTime;
    const entryExec = getExecType(t.entry_trigger || t.entryTrigger || '');
    const exitReason = t.exit_reason || t.exitReason || '';
    const exitExec = getExitExecType(t);
    // Shift C-type to next bar open
    const entryTime = shiftIfCType(rawEntryTime, entryExec);
    const exitTime = shiftIfCType(rawExitTime, exitExec);
    const entryMs = entryTime && entryTime !== '--' ? _safeMs(entryTime) : 0;
    const exitMs = exitTime && exitTime !== '--' ? _safeMs(exitTime) : 0;
    const rMult = t.r_multiple ?? t.rMultiple ?? t.pnlR ?? 0;

    if (entryMs >= firstBarTime && entryMs <= lastBarTime + tfMs) {
      m.push({
        time: entryTime, position: dir === 'LONG' ? 'belowBar' : 'aboveBar',
        shape: dir === 'LONG' ? 'arrowUp' : 'arrowDown',
        color: prefs.entryColor, text: prefs.showLabels ? 'Entry' : '', size: 1,
      });
    }
    if (exitMs >= firstBarTime && exitMs <= lastBarTime + tfMs) {
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

  // ---- Price-level cross markers (+) for entry AND exit ----
  const algoEntryData: any[] = [];
  const algoEntryMarkers: any[] = [];
  const seenEntry = new Set<string>();
  const algoExitData: any[] = [];
  const algoExitMarkers: any[] = [];
  const seenExit = new Set<string>();
  if (showTriggers) {
    const barTimestamps = bars.map((b: any) => b.timestamp);
    const snapToBar = (tradeTime: string): string | null => {
      const tradeMs = _safeMs(tradeTime);
      let bestTs = barTimestamps[0];
      let bestDist = Infinity;
      for (const ts of barTimestamps) {
        const dist = Math.abs(_safeMs(ts) - tradeMs);
        if (dist < bestDist) { bestDist = dist; bestTs = ts; }
      }
      return bestDist < 120000 ? bestTs : null;
    };

    for (const t of trades) {
      const rawEntryTime = t.entry_time || t.entryTime;
      const rawExitTime = t.exit_time || t.exitTime;
      const entryExec = getExecType(t.entry_trigger || t.entryTrigger || '');
      const exitReason = t.exit_reason || t.exitReason || '';
      const exitExec = getExitExecType(t);
      const entryTime = shiftIfCType(rawEntryTime, entryExec);
      const exitTime = shiftIfCType(rawExitTime, exitExec);
      const entryPrice = t.entry_price ?? t.entryPrice ?? 0;
      const exitPrice = t.exit_price ?? t.exitPrice ?? 0;
      const rMult = t.r_multiple ?? t.rMultiple ?? t.pnlR ?? 0;

      // Entry cross (+)
      if (entryTime && entryTime !== '--' && entryPrice > 0) {
        const snapped = snapToBar(entryTime);
        if (snapped && !seenEntry.has(snapped)) {
          seenEntry.add(snapped);
          algoEntryData.push({ time: snapped, value: entryPrice });
          algoEntryMarkers.push({ time: snapped, position: 'inBar', shape: 'cross', color: prefs.entryColor, text: '', size: 1 });
        }
      }
      // Exit cross (+)
      if (exitTime && exitTime !== '--' && exitPrice > 0) {
        const snapped = snapToBar(exitTime);
        if (snapped && !seenExit.has(snapped)) {
          seenExit.add(snapped);
          let color = rMult >= 0 ? prefs.exitWinColor : prefs.exitLossColor;
          if (exitReason === 'stop_loss') color = prefs.exitStopColor;
          else if (exitReason === 'bar_count_exit') color = prefs.exitBarCountColor;
          algoExitData.push({ time: snapped, value: exitPrice });
          algoExitMarkers.push({ time: snapped, position: 'inBar', shape: 'cross', color, text: '', size: 1 });
        }
      }
    }
  }

  // ---- Build panes ----
  const chartPanes: PaneConfig[] = [];

  // Pane 1: Confluence heatmap
  if (showConditions && heatmapConds.length > 0) {
    const n = heatmapConds.length;
    const hmSeries: SeriesConfig[] = heatmapConds.map((cond: any, idx: number) => {
      const paneVal = n - idx;
      const metColor = 'rgba(76,175,80,0.8)';
      const unmetColor = 'rgba(244,67,54,0.4)';
      // Detect cross-TF columns so the live-update effect knows which
      // state map to read from (primary states vs stateCrossTf).
      const isCrossTf = typeof cond.column === 'string' && cond.column.includes('__');
      return {
        type: 'Histogram' as const,
        data: bars.map((b: any, bi: number) => {
          const isPB = cond.fidelity === 'PB';
          const sourceBar = isPB && bi > 0 ? bars[bi - 1] : b;
          const stateVal = sourceBar[`_state_${cond.column}`];
          const isMet = stateVal != null && stateVal === cond.needed_state;
          return { time: b.timestamp, value: paneVal, color: isMet ? metColor : unmetColor };
        }),
        options: { priceLineVisible: false, lastValueVisible: false, title: cond.label },
        // Live-update metadata
        liveRole: 'heatmap_cell',
        stateColumn: cond.column,
        neededState: cond.needed_state,
        heatmapValue: paneVal,
        heatmapMetColor: metColor,
        heatmapUnmetColor: unmetColor,
        fidelity: cond.fidelity === 'PB' ? 'PB' : 'CB',
        crossTf: isCrossTf,
      };
    });
    chartPanes.push({ id: 'heatmap', height: Math.max(50, n * 20 + 10), series: hmSeries, hideTimeAxis: true });
  }

  // Pane 2: Price chart + overlays + trade markers
  const priceSeries: SeriesConfig[] = [
    {
      type: 'Candlestick',
      data: bars.map((b: any) => {
        const bar: any = { time: b.timestamp, open: b.open, high: b.high, low: b.low, close: b.close };
        // Apply candle color from indicator (e.g., Swing 1-2-3 pattern colors)
        if (candleColorColumn) {
          const indicatorColor = b[candleColorColumn];
          if (indicatorColor && typeof indicatorColor === 'string' && indicatorColor.startsWith('#')) {
            bar.color = indicatorColor;
            bar.borderColor = indicatorColor;
            bar.wickColor = indicatorColor;
          }
        }
        return bar;
      }),
      markers: tradeMarkers,
    },
  ];

  // ---- Alert (live) cross markers — xcross shape to distinguish from algo fills ----
  const alertEntryData: any[] = [];
  const alertEntryMarkers: any[] = [];
  const seenAlertEntry = new Set<string>();
  const alertExitData: any[] = [];
  const alertExitMarkers: any[] = [];
  const seenAlertExit = new Set<string>();
  if (showTriggers && alerts.length > 0) {
    const barTimestamps = bars.map((b: any) => b.timestamp);
    const snapToBar = (t: string): string | null => {
      const tms = _safeMs(t);
      let bestTs = barTimestamps[0];
      let bestDist = Infinity;
      for (const ts of barTimestamps) {
        const d = Math.abs(_safeMs(ts) - tms);
        if (d < bestDist) { bestDist = d; bestTs = ts; }
      }
      return bestDist < 120000 ? bestTs : null;
    };
    for (const a of alerts) {
      if (a.entryTime && a.entryTime !== '--' && a.entryPrice > 0) {
        const entryMs = _safeMs(a.entryTime);
        if (entryMs >= firstBarTime && entryMs <= lastBarTime) {
          const snapped = snapToBar(a.entryTime);
          if (snapped && !seenAlertEntry.has(snapped)) {
            seenAlertEntry.add(snapped);
            alertEntryData.push({ time: snapped, value: a.entryPrice });
            alertEntryMarkers.push({ time: snapped, position: 'inBar', shape: 'xcross', color: 'rgba(33,150,243,0.8)', text: '', size: 1 });
          }
        }
      }
      if (a.exitTime && a.exitTime !== '--' && a.exitPrice > 0) {
        const exitMs = _safeMs(a.exitTime);
        if (exitMs >= firstBarTime && exitMs <= lastBarTime) {
          const snapped = snapToBar(a.exitTime);
          if (snapped && !seenAlertExit.has(snapped)) {
            seenAlertExit.add(snapped);
            const reason = a.exitReason || '';
            let color: string;
            if (reason === 'stop' || reason === 'stop_loss') color = 'rgba(244,67,54,0.8)';
            else if (reason === 'bar_count_exit' || reason === 'max_hold_bars') color = 'rgba(38,166,154,0.8)';
            else if (reason === 'eod_exit' || reason === 'time_of_day_exit' || reason === 'session_exit') color = 'rgba(255,152,0,0.8)';
            else color = a.r != null && a.r >= 0 ? 'rgba(76,175,80,0.8)' : 'rgba(244,67,54,0.8)';
            alertExitData.push({ time: snapped, value: a.exitPrice });
            alertExitMarkers.push({ time: snapped, position: 'inBar', shape: 'xcross', color, text: '', size: 1 });
          }
        }
      }
    }
  }

  // Always push all 4 marker series so the price pane's series count stays
  // stable across re-renders (live alert refetch every 5s, forming bar every
  // 250ms). Empty data + empty markers is cheap; a changing series count
  // triggers full chart rebuild and wipes the user's zoom.
  priceSeries.push({
    type: 'Line', data: algoEntryData,
    options: { color: prefs.entryColor, lineVisible: false, pointMarkersVisible: false, priceLineVisible: false, crosshairMarkerVisible: false, lastValueVisible: false, title: '' },
    markers: algoEntryMarkers,
  });
  priceSeries.push({
    type: 'Line', data: algoExitData,
    options: { color: prefs.exitWinColor, lineVisible: false, pointMarkersVisible: false, priceLineVisible: false, crosshairMarkerVisible: false, lastValueVisible: false, title: '' },
    markers: algoExitMarkers,
  });
  priceSeries.push({
    type: 'Line', data: alertEntryData,
    options: { color: 'rgba(33,150,243,0.6)', lineVisible: false, pointMarkersVisible: false, priceLineVisible: false, crosshairMarkerVisible: false, lastValueVisible: false, title: '' },
    markers: alertEntryMarkers,
  });
  priceSeries.push({
    type: 'Line', data: alertExitData,
    options: { color: 'rgba(76,175,80,0.6)', lineVisible: false, pointMarkersVisible: false, priceLineVisible: false, crosshairMarkerVisible: false, lastValueVisible: false, title: '' },
    markers: alertExitMarkers,
  });

  // Overlay indicators (filtered to only plottable numeric columns)
  for (let i = 0; i < filteredOverlays.length; i++) {
    const col = filteredOverlays[i];
    priceSeries.push({
      type: 'Line',
      data: bars.filter((b: any) => b[col] != null).map((b: any) => ({ time: b.timestamp, value: b[col] })),
      options: { color: INDICATOR_COLORS[i % INDICATOR_COLORS.length], lineWidth: 2, title: col.replace(/_/g, ' ') },
      liveRole: 'overlay_line',
      indicatorColumn: col,
    });
  }
  chartPanes.push({ id: 'price', height: 350, series: priceSeries });

  // Pane 3+: Oscillators — group by indicator family so each indicator
  // (Stochastic, MACD, RSI, etc.) gets its own pane with proper scale.
  // Family is derived from the column prefix (stoch_*, macd_*, rsi_*, etc.).
  if (filteredOscs.length > 0) {
    const families = new Map<string, string[]>();
    for (const col of filteredOscs) {
      // Extract family prefix: split on underscore, take first segment (e.g. "stoch", "macd", "rsi")
      const family = col.split('_')[0] || col;
      if (!families.has(family)) families.set(family, []);
      families.get(family)!.push(col);
    }

    let paneIdx = 0;
    const familyEntries: [string, string[]][] = [];
    families.forEach((cols, family) => familyEntries.push([family, cols]));
    for (const [family, cols] of familyEntries) {
      const oscSeries: SeriesConfig[] = [];
      for (let i = 0; i < cols.length; i++) {
        const col = cols[i];
        const isHist = col.toLowerCase().includes('hist');
        oscSeries.push({
          type: isHist ? 'Histogram' : 'Line',
          data: bars.filter((b: any) => b[col] != null).map((b: any) => ({
            time: b.timestamp, value: b[col],
            ...(isHist ? { color: b[col] >= 0 ? 'rgba(76,175,80,0.6)' : 'rgba(244,67,54,0.6)' } : {}),
          })),
          options: isHist
            ? { priceLineVisible: false, lastValueVisible: false, title: col.replace(/_/g, ' ') }
            : { color: INDICATOR_COLORS[(paneIdx * 2 + i) % INDICATOR_COLORS.length], lineWidth: 1.5, priceLineVisible: false, title: col.replace(/_/g, ' ') },
          liveRole: isHist ? 'oscillator_hist' : 'oscillator_line',
          indicatorColumn: col,
        });
      }
      chartPanes.push({ id: `oscillator_${family}`, height: 120, series: oscSeries });
      paneIdx++;
    }
  }

  return chartPanes;
}
