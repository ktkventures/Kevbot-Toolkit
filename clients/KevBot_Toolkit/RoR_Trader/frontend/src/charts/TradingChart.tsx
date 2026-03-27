'use client';

import { useEffect, useRef, useCallback } from 'react';
import {
  createChart, CandlestickSeries, LineSeries,
  type IChartApi, type ISeriesApi, type CandlestickData, type LineData, type Time,
} from 'lightweight-charts';

export interface CandleData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

export interface OverlaySeries {
  name: string;
  data: { time: string; value: number }[];
  color: string;
  lineWidth?: number;
  lineStyle?: number;
}

export interface TradeMarker {
  time: string;
  position: 'aboveBar' | 'belowBar';
  shape: 'arrowUp' | 'arrowDown' | 'circle';
  color: string;
  text: string;
}

interface TradingChartProps {
  ohlcv: CandleData[];
  overlays?: OverlaySeries[];
  markers?: TradeMarker[];
  height?: number;
}

/**
 * TradingView lightweight-charts candlestick chart.
 *
 * Renders OHLCV data with optional indicator overlays and trade markers.
 * Uses imperative API via useRef + useEffect for full control.
 */

/** Convert ISO 8601 string or any date-like value to Unix seconds for LWC. */
function toUnixTime(t: string | number): number {
  if (typeof t === 'number') return t;
  return Math.floor(new Date(t).getTime() / 1000);
}

export default function TradingChart({
  ohlcv,
  overlays = [],
  markers = [],
  height = 400,
}: TradingChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<typeof CandlestickSeries> | null>(null);
  const overlaySeriesRefs = useRef<ISeriesApi<typeof LineSeries>[]>([]);

  // Get theme colors from CSS variables
  const getThemeColors = useCallback(() => {
    if (typeof window === 'undefined') return {
      bg: '#1E1E1E', text: '#DDD', grid: '#2B2B2B', border: '#333',
      up: '#4CAF50', down: '#f44336',
    };
    const style = getComputedStyle(document.documentElement);
    return {
      bg: style.getPropertyValue('--bg-primary').trim() || '#1E1E1E',
      text: style.getPropertyValue('--text-secondary').trim() || '#DDD',
      grid: style.getPropertyValue('--border').trim() || '#2B2B2B',
      border: style.getPropertyValue('--border').trim() || '#333',
      up: style.getPropertyValue('--green').trim() || '#4CAF50',
      down: style.getPropertyValue('--red').trim() || '#f44336',
    };
  }, []);

  useEffect(() => {
    if (!containerRef.current || ohlcv.length === 0) return;

    const colors = getThemeColors();

    // Create chart
    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height,
      layout: {
        background: { color: 'transparent' },
        textColor: colors.text,
      },
      grid: {
        vertLines: { color: colors.grid, style: 1 },
        horzLines: { color: colors.grid, style: 1 },
      },
      crosshair: { mode: 0 },
      rightPriceScale: { borderColor: colors.border },
      timeScale: { borderColor: colors.border },
    });
    chartRef.current = chart;

    // Add candlestick series (v5 API: addSeries with series type)
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: colors.up,
      downColor: colors.down,
      borderUpColor: colors.up,
      borderDownColor: colors.down,
      wickUpColor: colors.up,
      wickDownColor: colors.down,
    });

    const candleData: CandlestickData[] = ohlcv
      .map((c) => ({
        time: toUnixTime(c.time) as Time,
        open: Number(c.open),
        high: Number(c.high),
        low: Number(c.low),
        close: Number(c.close),
      }))
      .filter((c) => isFinite(c.time as number) && isFinite(c.open) && isFinite(c.high) && isFinite(c.low) && isFinite(c.close))
      .sort((a, b) => (a.time as number) - (b.time as number));
    if (candleData.length === 0) { chart.remove(); return; }
    candleSeries.setData(candleData);
    candleSeriesRef.current = candleSeries;

    // Markers: LWC v5 removed setMarkers() from series API.
    // Use try-catch as defensive fallback for any version differences.
    if (markers.length > 0) {
      try {
        const lwcMarkers = markers
          .filter((m) => m.time)
          .map((m) => ({
            time: toUnixTime(m.time) as Time,
            position: m.position,
            shape: m.shape,
            color: m.color,
            text: m.text,
          }))
          .filter((m) => isFinite(m.time as number))
          .sort((a, b) => (a.time as number) - (b.time as number));
        if (typeof (candleSeries as any).setMarkers === 'function') {
          (candleSeries as any).setMarkers(lwcMarkers);
        }
      } catch {
        // v5 may not support setMarkers — silently skip
      }
    }

    // Add overlay line series
    const overlayRefs: ISeriesApi<'Line'>[] = [];
    for (const overlay of overlays) {
      const lineSeries = chart.addSeries(LineSeries, {
        color: overlay.color,
        lineWidth: (overlay.lineWidth || 1) as 1 | 2 | 3 | 4,
        lineStyle: overlay.lineStyle || 0,
        priceLineVisible: false,
        lastValueVisible: false,
      });
      const lineData: LineData[] = overlay.data
        .map((d) => ({
          time: toUnixTime(d.time) as Time,
          value: d.value,
        }))
        .filter((d) => isFinite(d.time as number) && isFinite(d.value))
        .sort((a, b) => (a.time as number) - (b.time as number));
      lineSeries.setData(lineData);
      overlayRefs.push(lineSeries);
    }
    overlaySeriesRefs.current = overlayRefs;

    // Fit content
    chart.timeScale().fitContent();

    // Resize handler
    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
      overlaySeriesRefs.current = [];
    };
  }, [ohlcv, overlays, markers, height, getThemeColors]);

  if (ohlcv.length === 0) {
    return (
      <div
        style={{
          height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--text-secondary)',
          fontSize: '13px',
          background: 'var(--bg-secondary)',
          borderRadius: '8px',
        }}
      >
        No chart data
      </div>
    );
  }

  return <div ref={containerRef} style={{ width: '100%', height }} />;
}
