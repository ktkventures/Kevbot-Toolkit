'use client';

/**
 * SyncedChartPane — renders multiple synchronized lightweight-charts panes.
 *
 * M8.5 Phase B+ refactor: chart instance + series objects PERSIST across
 * re-renders. The setup useEffect is keyed by `paneStructureKey` (a stable
 * hash of pane id/height/series-types) so it only fires when the structure
 * genuinely changes. A separate data effect runs on every `panes` prop
 * change and pushes new data via `series.setData()` / `series.setMarkers()`
 * without recreating the chart. Visible time range is saved before any
 * structure rebuild and restored after, so the user's zoom/scroll position
 * survives toggling indicators / changing candle count.
 *
 * The forming-bar imperative update (M8.5 Phase B) is unchanged.
 */

import { useEffect, useMemo, useRef, useCallback } from 'react';
import {
  createChart,
  type IChartApi, type ISeriesApi, type SeriesType, type Time,
} from 'lightweight-charts';
import { SessionHighlighting } from './plugins/SessionHighlighting';

/** Convert ISO 8601 string or number to Unix seconds. */
function toUnixTime(t: string | number): number {
  if (typeof t === 'number') return t;
  return Math.floor(new Date(t).getTime() / 1000);
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SeriesConfig {
  type: 'Candlestick' | 'Line' | 'Histogram';
  data: Record<string, any>[];
  options?: Record<string, any>;
  markers?: Record<string, any>[];
}

export interface PrimitiveConfig {
  type: 'sessionHighlighting';
  seriesIndex?: number;
  options: any;
}

export interface PaneConfig {
  id: string;
  height: number;
  series: SeriesConfig[];
  primitives?: PrimitiveConfig[];
  /** Hide time axis on this pane (for heatmap/middle panes) */
  hideTimeAxis?: boolean;
}

export interface LiveCandle {
  time: string | number;
  open: number;
  high: number;
  low: number;
  close: number;
}

interface SyncedChartPaneProps {
  panes: PaneConfig[];
  /** Candle theme colors */
  upColor?: string;
  downColor?: string;
  upBorderColor?: string;
  gridLines?: boolean;
  /** Number of empty bars to show to the right of the last candle */
  rightOffset?: number;
  /**
   * M8.5: latest forming/completed bar pushed imperatively to the primary
   * pane's candlestick series via series.update(). Does NOT trigger a
   * re-render — the update runs in a dedicated useEffect that only
   * depends on this prop.
   */
  formingBar?: LiveCandle | null;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Transform raw series data points to LWC format. */
function transformSeriesData(seriesCfg: SeriesConfig): any[] {
  return seriesCfg.data
    .map((d: any) => {
      const time = toUnixTime(d.time ?? d.timestamp) as Time;
      if (!isFinite(time as number)) return null;
      if (seriesCfg.type === 'Candlestick') {
        if (!isFinite(d.open) || !isFinite(d.high) || !isFinite(d.low) || !isFinite(d.close)) return null;
        const candle: any = {
          time, open: Number(d.open), high: Number(d.high),
          low: Number(d.low), close: Number(d.close),
        };
        if (d.color) candle.color = d.color;
        if (d.borderColor) candle.borderColor = d.borderColor;
        if (d.wickColor) candle.wickColor = d.wickColor;
        return candle;
      }
      if (!isFinite(d.value)) return null;
      return d.color ? { time, value: Number(d.value), color: d.color }
                     : { time, value: Number(d.value) };
    })
    .filter(Boolean)
    .sort((a: any, b: any) => (a.time as number) - (b.time as number));
}

/** Transform marker timestamps to LWC format and sort. */
function transformMarkers(markers: Record<string, any>[]): any[] {
  return markers
    .map((m: any) => ({ ...m, time: toUnixTime(m.time) as Time }))
    .filter((m: any) => isFinite(m.time as number))
    .sort((a: any, b: any) => (a.time as number) - (b.time as number));
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function SyncedChartPane({
  panes,
  upColor,
  downColor,
  upBorderColor,
  gridLines = true,
  rightOffset = 3,
  formingBar = null,
}: SyncedChartPaneProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartsRef = useRef<IChartApi[]>([]);
  const seriesRef = useRef<ISeriesApi<SeriesType>[][]>([]); // [paneIdx][seriesIdx]
  const syncingRef = useRef(false);
  const primaryCandleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  // Preserves user's zoom/scroll across rare structure rebuilds (toggle
  // Show Conditions, candle-count change beyond visible range, etc.)
  const lastVisibleRangeRef = useRef<{ from: Time; to: Time } | null>(null);
  // Tracks whether the data effect has run at least once for the current
  // chart instance. Reset by the structure effect on (re)create. Only
  // when false do we call fitContent on data load.
  const hasRenderedInitialDataRef = useRef(false);

  const getThemeColors = useCallback(() => {
    if (typeof window === 'undefined') return { text: '#DDD', grid: '#2B2B2B', border: '#333', up: '#4CAF50', down: '#f44336' };
    const s = getComputedStyle(document.documentElement);
    return {
      text: s.getPropertyValue('--text-secondary').trim() || '#DDD',
      grid: s.getPropertyValue('--border').trim() || '#2B2B2B',
      border: s.getPropertyValue('--border').trim() || '#333',
      up: s.getPropertyValue('--green').trim() || '#4CAF50',
      down: s.getPropertyValue('--red').trim() || '#f44336',
    };
  }, []);

  // M8.5 B+: stable structure hash. Changes only when pane count / heights /
  // series types / primitives genuinely change — NOT when only data changes.
  // Excludes data so live updates don't fire structure rebuilds.
  const paneStructureKey = useMemo(
    () => JSON.stringify(panes.map(p => ({
      id: p.id,
      height: p.height,
      hideTimeAxis: p.hideTimeAxis,
      seriesTypes: p.series.map(s => s.type),
      primitives: (p.primitives || []).map(pr => pr.type),
    }))),
    [panes],
  );

  // ---- STRUCTURE EFFECT ----
  // Creates chart instances + adds empty series. Runs only when structure
  // changes (rare) or theme colors change. Restores visible range from prior
  // chart so the user's zoom/scroll position survives the rebuild.
  useEffect(() => {
    if (!containerRef.current || panes.length === 0) return;

    const colors = getThemeColors();
    const up = upColor || colors.up;
    const down = downColor || colors.down;
    const borderUp = upBorderColor || up;
    const charts: IChartApi[] = [];
    const allSeries: ISeriesApi<SeriesType>[][] = [];
    let primaryCandle: ISeriesApi<'Candlestick'> | null = null;

    // Save scroll position before tearing down old charts
    if (chartsRef.current[0]) {
      try {
        const r = chartsRef.current[0].timeScale().getVisibleRange();
        if (r) lastVisibleRangeRef.current = r as any;
      } catch { /* ignore */ }
    }

    // Tear down any prior charts first (effect re-runs replace, not append)
    for (const c of chartsRef.current) {
      try { c.remove(); } catch { /* ignore */ }
    }
    chartsRef.current = [];
    seriesRef.current = [];
    primaryCandleSeriesRef.current = null;
    // The new chart hasn't received data yet — flag it so the next data
    // effect fitContents (only on first load per chart instance).
    hasRenderedInitialDataRef.current = false;

    const container = containerRef.current;
    container.innerHTML = '';

    for (let pi = 0; pi < panes.length; pi++) {
      const pane = panes[pi];
      const div = document.createElement('div');
      div.style.width = '100%';
      container.appendChild(div);

      const chart = createChart(div, {
        width: container.clientWidth,
        height: pane.height,
        layout: { background: { color: 'transparent' }, textColor: colors.text },
        grid: gridLines
          ? { vertLines: { color: colors.grid, style: 1 }, horzLines: { color: colors.grid, style: 1 } }
          : { vertLines: { visible: false }, horzLines: { visible: false } },
        crosshair: { mode: 0 },
        rightPriceScale: { borderColor: colors.border },
        timeScale: {
          borderColor: colors.border,
          timeVisible: true,
          secondsVisible: true,
          visible: !pane.hideTimeAxis,
          rightOffset: rightOffset,
        },
      });
      charts.push(chart);

      // Create empty series instances. Data is pushed by the data effect.
      const paneSeries: ISeriesApi<SeriesType>[] = [];
      for (const seriesCfg of pane.series) {
        let chartSeries: ISeriesApi<SeriesType> | undefined;
        switch (seriesCfg.type) {
          case 'Candlestick':
            chartSeries = chart.addCandlestickSeries({
              upColor: up, downColor: down,
              borderUpColor: borderUp, borderDownColor: down,
              wickUpColor: borderUp, wickDownColor: down,
              ...seriesCfg.options,
            });
            if (pi === 0 && !primaryCandle) {
              primaryCandle = chartSeries as ISeriesApi<'Candlestick'>;
            }
            break;
          case 'Line':
            chartSeries = chart.addLineSeries({
              priceLineVisible: false, lastValueVisible: false,
              ...seriesCfg.options,
            });
            break;
          case 'Histogram':
            chartSeries = chart.addHistogramSeries({
              priceLineVisible: false,
              ...seriesCfg.options,
            });
            break;
          default:
            continue;
        }
        paneSeries.push(chartSeries);

        // Price lines (set once on creation; rare)
        if ((seriesCfg as any).priceLines && chartSeries) {
          try {
            for (const pl of (seriesCfg as any).priceLines) {
              (chartSeries as any).createPriceLine({
                price: pl.price,
                color: pl.color || 'rgba(255,255,255,0.5)',
                lineWidth: pl.lineWidth || 1,
                lineStyle: pl.lineStyle || 2,
                axisLabelVisible: pl.axisLabelVisible !== false,
                title: pl.title || '',
              });
            }
          } catch (e) {
            console.warn('createPriceLine failed:', e);
          }
        }
      }
      allSeries.push(paneSeries);

      // Attach primitives
      if (pane.primitives) {
        for (const prim of pane.primitives) {
          const target = paneSeries[prim.seriesIndex || 0];
          if (!target) continue;
          try {
            if (prim.type === 'sessionHighlighting') {
              (target as any).attachPrimitive(new SessionHighlighting(prim.options));
            }
          } catch (e) {
            console.warn('Primitive attach failed:', e);
          }
        }
      }
    }

    chartsRef.current = charts;
    seriesRef.current = allSeries;
    primaryCandleSeriesRef.current = primaryCandle;

    // ---- Cross-pane synchronization ----
    if (charts.length > 1) {
      for (const chart of charts) {
        chart.timeScale().subscribeVisibleTimeRangeChange((range) => {
          if (syncingRef.current || !range) return;
          syncingRef.current = true;
          for (const other of charts) {
            if (other !== chart) {
              try { other.timeScale().setVisibleRange(range); } catch { /* ignore */ }
            }
          }
          syncingRef.current = false;
          // Track latest visible range so we can restore across rebuilds
          lastVisibleRangeRef.current = range as any;
        });
      }
    } else if (charts[0]) {
      charts[0].timeScale().subscribeVisibleTimeRangeChange((range) => {
        if (range) lastVisibleRangeRef.current = range as any;
      });
    }

    // Resize handler
    const handleResize = () => {
      if (!containerRef.current) return;
      const w = containerRef.current.clientWidth;
      for (const c of charts) c.applyOptions({ width: w });
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      for (const c of charts) {
        try { c.remove(); } catch { /* ignore */ }
      }
      chartsRef.current = [];
      seriesRef.current = [];
      primaryCandleSeriesRef.current = null;
    };
  }, [paneStructureKey, upColor, downColor, upBorderColor, gridLines, rightOffset, getThemeColors]);

  // ---- DATA EFFECT ----
  // Runs whenever panes prop changes. Pushes data to existing series via
  // setData / setMarkers — does NOT recreate the chart, so user's zoom/scroll
  // is preserved naturally (LWC v4: setData does not reset visible range;
  // chart auto-tracks the rightmost edge if user is at it, otherwise stays
  // wherever the user panned to).
  //
  // IMPORTANT: do NOT call setVisibleRange here. Doing so would force the
  // view to a snapshotted range every time data updates, overriding LWC's
  // natural "follow rightmost / preserve scroll" behavior. Visible-range
  // restore is only needed across STRUCTURE rebuilds (handled in the
  // structure effect).
  useEffect(() => {
    const charts = chartsRef.current;
    const allSeries = seriesRef.current;
    const wasFirstLoad = !hasRenderedInitialDataRef.current;
    if (charts.length === 0 || allSeries.length !== panes.length) return;

    for (let pi = 0; pi < panes.length; pi++) {
      const pane = panes[pi];
      const paneSeries = allSeries[pi];
      if (!paneSeries || paneSeries.length !== pane.series.length) continue;
      for (let si = 0; si < pane.series.length; si++) {
        const cfg = pane.series[si];
        const series = paneSeries[si];
        if (!series) continue;
        // Always call setData (even with empty array) so prior data is
        // cleared correctly when a series becomes empty.
        const data = transformSeriesData(cfg);
        try { series.setData(data); } catch (e) {
          console.warn('setData failed:', e);
        }
        // Always call setMarkers (even with empty array) so prior markers
        // are cleared when alerts/trades roll off the visible window.
        try {
          const valid = cfg.markers ? transformMarkers(cfg.markers) : [];
          (series as any).setMarkers(valid);
        } catch (e) {
          console.warn('setMarkers failed:', e);
        }
      }
    }

    // First-time data load for this chart instance: either restore the
    // user's prior visible range (if we saved one before a structure
    // rebuild) or fitContent (truly fresh chart). Subsequent data updates:
    // do nothing — LWC preserves the user's view + auto-tracks the right
    // edge naturally.
    if (wasFirstLoad && charts[0]) {
      const savedRange = lastVisibleRangeRef.current;
      if (savedRange) {
        try { charts[0].timeScale().setVisibleRange(savedRange); }
        catch { try { charts[0].timeScale().fitContent(); } catch { /* ignore */ } }
      } else {
        try { charts[0].timeScale().fitContent(); } catch { /* ignore */ }
      }
      hasRenderedInitialDataRef.current = true;
    }
  }, [panes]);

  // ---- LIVE FORMING-BAR EFFECT (M8.5 Phase B, unchanged) ----
  useEffect(() => {
    const series = primaryCandleSeriesRef.current;
    if (!series || !formingBar) return;
    const t = toUnixTime(formingBar.time);
    if (!isFinite(t)) return;
    if (!isFinite(formingBar.open) || !isFinite(formingBar.high) ||
        !isFinite(formingBar.low) || !isFinite(formingBar.close)) return;
    try {
      series.update({
        time: t as Time,
        open: Number(formingBar.open),
        high: Number(formingBar.high),
        low: Number(formingBar.low),
        close: Number(formingBar.close),
      });
    } catch {
      // LWC rejects updates with time earlier than last bar — drop silently.
    }
  }, [formingBar]);

  if (panes.length === 0) {
    return (
      <div style={{ height: 400, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)', fontSize: '13px' }}>
        No chart data
      </div>
    );
  }

  return <div ref={containerRef} style={{ width: '100%' }} />;
}
