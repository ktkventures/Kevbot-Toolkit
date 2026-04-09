'use client';

/**
 * SyncedChartPane — renders multiple synchronized lightweight-charts panes.
 *
 * Ported from streamlit_lwc_fork/LightweightCharts.tsx. Each pane gets its own
 * createChart() instance. Zoom/scroll is synchronized across all panes via
 * subscribeVisibleLogicalRangeChange + subscribeVisibleTimeRangeChange.
 *
 * Supports: Candlestick, Line, Histogram series + SessionHighlighting primitives.
 */

import { useEffect, useRef, useCallback } from 'react';
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

interface SyncedChartPaneProps {
  panes: PaneConfig[];
  /** Candle theme colors */
  upColor?: string;
  downColor?: string;
  upBorderColor?: string;
  gridLines?: boolean;
  /** Number of empty bars to show to the right of the last candle */
  rightOffset?: number;
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
}: SyncedChartPaneProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartsRef = useRef<IChartApi[]>([]);
  const syncingRef = useRef(false); // prevent infinite sync loops

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

  useEffect(() => {
    if (!containerRef.current || panes.length === 0) return;

    const colors = getThemeColors();
    const up = upColor || colors.up;
    const down = downColor || colors.down;
    const borderUp = upBorderColor || up;
    const charts: IChartApi[] = [];

    // Create container divs for each pane
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

      // Create series (LWC v4 API)
      const createdSeries: ISeriesApi<SeriesType>[] = [];
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
            break;
          case 'Line':
            chartSeries = chart.addLineSeries({
              priceLineVisible: false,
              lastValueVisible: false,
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

        // Transform and set data
        const data = seriesCfg.data
          .map((d: any) => {
            const time = toUnixTime(d.time ?? d.timestamp) as Time;
            if (!isFinite(time as number)) return null;
            if (seriesCfg.type === 'Candlestick') {
              if (!isFinite(d.open) || !isFinite(d.high) || !isFinite(d.low) || !isFinite(d.close)) return null;
              const candle: any = { time, open: Number(d.open), high: Number(d.high), low: Number(d.low), close: Number(d.close) };
              // Preserve per-bar color overrides (e.g., indicator candle coloring)
              if (d.color) candle.color = d.color;
              if (d.borderColor) candle.borderColor = d.borderColor;
              if (d.wickColor) candle.wickColor = d.wickColor;
              return candle;
            }
            if (!isFinite(d.value)) return null;
            return d.color ? { time, value: Number(d.value), color: d.color } : { time, value: Number(d.value) };
          })
          .filter(Boolean)
          .sort((a: any, b: any) => (a.time as number) - (b.time as number));

        if (data.length > 0) chartSeries.setData(data);

        // Markers (v4 native API)
        if (seriesCfg.markers && seriesCfg.markers.length > 0 && chartSeries) {
          try {
            const validMarkers = seriesCfg.markers
              .map((m: any) => ({ ...m, time: toUnixTime(m.time) as Time }))
              .filter((m: any) => isFinite(m.time as number))
              .sort((a: any, b: any) => (a.time as number) - (b.time as number));
            if (validMarkers.length > 0) {
              chartSeries.setMarkers(validMarkers);
            }
          } catch (e) {
            console.warn('setMarkers failed:', e);
          }
        }

        // Price lines (horizontal reference lines for stop/target/entry)
        if ((seriesCfg as any).priceLines && chartSeries) {
          try {
            for (const pl of (seriesCfg as any).priceLines) {
              (chartSeries as any).createPriceLine({
                price: pl.price,
                color: pl.color || 'rgba(255,255,255,0.5)',
                lineWidth: pl.lineWidth || 1,
                lineStyle: pl.lineStyle || 2, // 0=solid, 1=dotted, 2=dashed
                axisLabelVisible: pl.axisLabelVisible !== false,
                title: pl.title || '',
              });
            }
          } catch (e) {
            console.warn('createPriceLine failed:', e);
          }
        }

        createdSeries.push(chartSeries);
      }

      // Attach primitives
      if (pane.primitives) {
        for (const prim of pane.primitives) {
          const target = createdSeries[prim.seriesIndex || 0];
          if (!target) continue;
          try {
            if (prim.type === 'sessionHighlighting') {
              target.attachPrimitive(new SessionHighlighting(prim.options));
            }
          } catch (e) {
            console.warn('Primitive attach failed:', e);
          }
        }
      }

      // Don't fitContent per-chart yet — do it once after sync is set up
    }

    chartsRef.current = charts;

    // ---- Cross-pane synchronization (time-based) ----
    // Use visible time range instead of logical range to avoid index misalignment
    // when panes have different numbers of data points (e.g., oscillator warmup).
    if (charts.length > 1) {
      for (const chart of charts) {
        chart.timeScale().subscribeVisibleTimeRangeChange((range) => {
          if (syncingRef.current || !range) return;
          syncingRef.current = true;
          for (const other of charts) {
            if (other !== chart) {
              try {
                other.timeScale().setVisibleRange(range);
              } catch { /* ignore if range is invalid for this chart */ }
            }
          }
          syncingRef.current = false;
        });
      }
    }

    // Fit all charts to content independently, then sync via time range
    for (const chart of charts) {
      chart.timeScale().fitContent();
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
      for (const c of charts) c.remove();
      chartsRef.current = [];
    };
  }, [panes, upColor, downColor, upBorderColor, gridLines, rightOffset, getThemeColors]);

  if (panes.length === 0) {
    return (
      <div style={{ height: 400, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)', fontSize: '13px' }}>
        No chart data
      </div>
    );
  }

  return <div ref={containerRef} style={{ width: '100%' }} />;
}
