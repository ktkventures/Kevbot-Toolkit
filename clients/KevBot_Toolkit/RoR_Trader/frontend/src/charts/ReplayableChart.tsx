'use client';

/**
 * ReplayableChart — LWC wrapper designed for imperative data updates.
 *
 * Unlike SyncedChartPane (which destroys/recreates on every panes change),
 * this component creates the chart ONCE and exposes setSeriesData() / setSeriesMarkers()
 * via forwardRef + useImperativeHandle for flicker-free updates.
 */

import {
  useEffect, useRef, useCallback, useImperativeHandle, forwardRef,
} from 'react';
import {
  createChart,
  type IChartApi, type ISeriesApi, type SeriesType, type Time,
} from 'lightweight-charts';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SeriesSetup {
  type: 'Candlestick' | 'Line' | 'Histogram';
  options?: Record<string, any>;
  priceLines?: {
    price: number;
    color: string;
    lineWidth?: number;
    lineStyle?: number; // 0=solid, 1=dotted, 2=dashed
    axisLabelVisible?: boolean;
    title?: string;
  }[];
}

export interface ReplayableChartHandle {
  setSeriesData(seriesIndex: number, data: any[]): void;
  setSeriesMarkers(seriesIndex: number, markers: any[]): void;
  fitContent(): void;
}

interface ReplayableChartProps {
  id: string;
  height: number;
  seriesSetup: SeriesSetup[];
  upColor?: string;
  downColor?: string;
  upBorderColor?: string;
  gridLines?: boolean;
  rightOffset?: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function toUnixTime(t: string | number): number {
  if (typeof t === 'number') return t;
  return Math.floor(new Date(t).getTime() / 1000);
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const ReplayableChart = forwardRef<ReplayableChartHandle, ReplayableChartProps>(
  function ReplayableChart(
    { id, height, seriesSetup, upColor, downColor, upBorderColor, gridLines = true, rightOffset = 3 },
    ref,
  ) {
    const containerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const seriesRef = useRef<ISeriesApi<SeriesType>[]>([]);

    const getThemeColors = useCallback(() => {
      if (typeof window === 'undefined') {
        return { text: '#DDD', grid: '#2B2B2B', border: '#333', up: '#4CAF50', down: '#f44336' };
      }
      const s = getComputedStyle(document.documentElement);
      return {
        text: s.getPropertyValue('--text-secondary').trim() || '#DDD',
        grid: s.getPropertyValue('--border').trim() || '#2B2B2B',
        border: s.getPropertyValue('--border').trim() || '#333',
        up: s.getPropertyValue('--green').trim() || '#4CAF50',
        down: s.getPropertyValue('--red').trim() || '#f44336',
      };
    }, []);

    // Create chart + series ONCE
    useEffect(() => {
      if (!containerRef.current) return;
      const colors = getThemeColors();
      const up = upColor || colors.up;
      const down = downColor || colors.down;
      const borderUp = upBorderColor || up;

      const chart = createChart(containerRef.current, {
        width: containerRef.current.clientWidth,
        height,
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
          rightOffset,
        },
      });

      chartRef.current = chart;
      const created: ISeriesApi<SeriesType>[] = [];

      for (const setup of seriesSetup) {
        let series: ISeriesApi<SeriesType> | undefined;
        switch (setup.type) {
          case 'Candlestick':
            series = chart.addCandlestickSeries({
              upColor: up, downColor: down,
              borderUpColor: borderUp, borderDownColor: down,
              wickUpColor: borderUp, wickDownColor: down,
              ...setup.options,
            });
            break;
          case 'Line':
            series = chart.addLineSeries({
              priceLineVisible: false,
              lastValueVisible: false,
              ...setup.options,
            });
            break;
          case 'Histogram':
            series = chart.addHistogramSeries({
              priceLineVisible: false,
              ...setup.options,
            });
            break;
        }
        if (!series) continue;

        // Create static price lines
        if (setup.priceLines) {
          for (const pl of setup.priceLines) {
            try {
              (series as any).createPriceLine({
                price: pl.price,
                color: pl.color || 'rgba(255,255,255,0.5)',
                lineWidth: pl.lineWidth || 1,
                lineStyle: pl.lineStyle ?? 2,
                axisLabelVisible: pl.axisLabelVisible !== false,
                title: pl.title || '',
              });
            } catch { /* ignore */ }
          }
        }

        created.push(series);
      }

      seriesRef.current = created;
      chart.timeScale().fitContent();

      // Resize
      const ro = new ResizeObserver(() => {
        if (containerRef.current) {
          chart.applyOptions({ width: containerRef.current.clientWidth });
        }
      });
      ro.observe(containerRef.current);

      return () => {
        ro.disconnect();
        chart.remove();
        chartRef.current = null;
        seriesRef.current = [];
      };
      // Only re-create on id change (structural change), NOT on data changes
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [id]);

    // Expose imperative methods
    useImperativeHandle(ref, () => ({
      setSeriesData(seriesIndex: number, data: any[]) {
        const series = seriesRef.current[seriesIndex];
        if (!series) return;
        const transformed = data
          .map((d: any) => {
            const time = toUnixTime(d.time ?? d.timestamp) as Time;
            if (!isFinite(time as number)) return null;
            if (seriesSetup[seriesIndex]?.type === 'Candlestick') {
              if (!isFinite(d.open) || !isFinite(d.high) || !isFinite(d.low) || !isFinite(d.close)) return null;
              const bar: any = { time, open: Number(d.open), high: Number(d.high), low: Number(d.low), close: Number(d.close) };
              // Per-bar color overrides (for forming candle)
              if (d.color) bar.color = d.color;
              if (d.borderColor) bar.borderColor = d.borderColor;
              if (d.wickColor) bar.wickColor = d.wickColor;
              return bar;
            }
            if (!isFinite(d.value)) return null;
            return d.color ? { time, value: Number(d.value), color: d.color } : { time, value: Number(d.value) };
          })
          .filter(Boolean)
          .sort((a: any, b: any) => (a.time as number) - (b.time as number));
        try {
          series.setData(transformed);
        } catch (e) {
          console.warn('ReplayableChart setData failed:', e);
        }
      },

      setSeriesMarkers(seriesIndex: number, markers: any[]) {
        const series = seriesRef.current[seriesIndex];
        if (!series) return;
        try {
          const valid = markers
            .map((m: any) => ({ ...m, time: toUnixTime(m.time) as Time }))
            .filter((m: any) => isFinite(m.time as number))
            .sort((a: any, b: any) => (a.time as number) - (b.time as number));
          series.setMarkers(valid);
        } catch (e) {
          console.warn('ReplayableChart setMarkers failed:', e);
        }
      },

      fitContent() {
        chartRef.current?.timeScale().fitContent();
      },
    }), [seriesSetup]);

    return <div ref={containerRef} style={{ width: '100%', minHeight: height }} />;
  },
);

export default ReplayableChart;
