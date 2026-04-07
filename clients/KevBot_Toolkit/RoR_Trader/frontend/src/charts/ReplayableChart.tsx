'use client';

/**
 * ReplayableChart — LWC wrapper designed for prop-driven data updates.
 *
 * Unlike SyncedChartPane (which destroys/recreates on every panes change),
 * this component creates the chart ONCE and updates data via setData()
 * when the `seriesData` prop changes — flicker-free.
 *
 * No refs needed: data flows through props, avoiding Next.js dynamic() ref issues.
 */

import { useEffect, useRef, useCallback } from 'react';
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

export interface SeriesDataInput {
  data: any[];
  markers?: any[];
}

interface ReplayableChartProps {
  id: string;
  height: number;
  seriesSetup: SeriesSetup[];
  /** Per-series data + markers — update this prop to push new data */
  seriesData?: SeriesDataInput[];
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

function transformCandleData(data: any[]): any[] {
  return data
    .map((d: any) => {
      const time = toUnixTime(d.time ?? d.timestamp) as Time;
      if (!isFinite(time as number)) return null;
      if (!isFinite(d.open) || !isFinite(d.high) || !isFinite(d.low) || !isFinite(d.close)) return null;
      const bar: any = { time, open: Number(d.open), high: Number(d.high), low: Number(d.low), close: Number(d.close) };
      if (d.color) bar.color = d.color;
      if (d.borderColor) bar.borderColor = d.borderColor;
      if (d.wickColor) bar.wickColor = d.wickColor;
      return bar;
    })
    .filter(Boolean)
    .sort((a: any, b: any) => (a.time as number) - (b.time as number));
}

function transformLineData(data: any[]): any[] {
  return data
    .map((d: any) => {
      const time = toUnixTime(d.time ?? d.timestamp) as Time;
      if (!isFinite(time as number) || !isFinite(d.value)) return null;
      return d.color ? { time, value: Number(d.value), color: d.color } : { time, value: Number(d.value) };
    })
    .filter(Boolean)
    .sort((a: any, b: any) => (a.time as number) - (b.time as number));
}

function transformMarkers(markers: any[]): any[] {
  return markers
    .map((m: any) => ({ ...m, time: toUnixTime(m.time) as Time }))
    .filter((m: any) => isFinite(m.time as number))
    .sort((a: any, b: any) => (a.time as number) - (b.time as number));
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function ReplayableChart({
  id, height, seriesSetup, seriesData,
  upColor, downColor, upBorderColor, gridLines = true, rightOffset = 3,
}: ReplayableChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<SeriesType>[]>([]);
  const setupRef = useRef<SeriesSetup[]>(seriesSetup);
  setupRef.current = seriesSetup;

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

  // Push data to series whenever seriesData prop changes
  useEffect(() => {
    if (!seriesData || seriesRef.current.length === 0) return;

    for (let i = 0; i < seriesData.length && i < seriesRef.current.length; i++) {
      const sd = seriesData[i];
      const series = seriesRef.current[i];
      const setup = setupRef.current[i];
      if (!sd || !series) continue;

      try {
        if (sd.data.length > 0) {
          const transformed = setup?.type === 'Candlestick'
            ? transformCandleData(sd.data)
            : transformLineData(sd.data);
          series.setData(transformed);
        } else {
          series.setData([]);
        }
      } catch (e) {
        console.warn('ReplayableChart setData failed:', e);
      }

      try {
        if (sd.markers && sd.markers.length > 0) {
          series.setMarkers(transformMarkers(sd.markers));
        } else {
          series.setMarkers([]);
        }
      } catch (e) {
        console.warn('ReplayableChart setMarkers failed:', e);
      }
    }

    chartRef.current?.timeScale().fitContent();
  }, [seriesData]);

  return <div ref={containerRef} style={{ width: '100%', minHeight: height }} />;
}
