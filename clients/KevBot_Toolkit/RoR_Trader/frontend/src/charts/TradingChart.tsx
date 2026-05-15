'use client';

import { useEffect, useRef, useCallback } from 'react';
import {
  createChart,
  type IChartApi, type ISeriesApi, type CandlestickData, type LineData, type Time, type SeriesType,
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
  position: 'aboveBar' | 'belowBar' | 'inBar';
  shape: 'arrowUp' | 'arrowDown' | 'circle' | 'square';
  color: string;
  text: string;
}

interface TradingChartProps {
  ohlcv: CandleData[];
  overlays?: OverlaySeries[];
  markers?: TradeMarker[];
  height?: number;
  // Display preferences
  upColor?: string;
  downColor?: string;
  upBorderColor?: string;
  gridLines?: boolean;
  rightOffset?: number;
  timeVisible?: boolean;
  secondsVisible?: boolean;
  // M8.5: optional live-bar update applied imperatively via series.update()
  // so that forming/completed bar pushes from Ralph don't trigger React
  // re-renders. Prop change → single imperative call, zero cascade.
  formingBar?: CandleData | null;
  // Phase F.1 (2026-05-14): explicit visible time range (ISO strings).
  // When provided, overrides the default fitContent() so multiple charts
  // sharing the same range display the same x-axis even when their data
  // extents differ (e.g., side-by-side parity-comparison panes).
  visibleRange?: { from: string; to: string } | null;
}

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
  upColor,
  downColor,
  upBorderColor,
  gridLines = true,
  rightOffset = 3,
  timeVisible = true,
  secondsVisible = true,
  formingBar = null,
  visibleRange = null,
}: TradingChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const overlaySeriesRefs = useRef<ISeriesApi<'Line'>[]>([]);

  // Get theme colors from CSS variables (fallback when no explicit colors passed)
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
    const up = upColor || colors.up;
    const down = downColor || colors.down;
    const borderUp = upBorderColor || up;

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height,
      layout: {
        background: { color: 'transparent' },
        textColor: colors.text,
      },
      grid: gridLines ? {
        vertLines: { color: colors.grid, style: 1 },
        horzLines: { color: colors.grid, style: 1 },
      } : {
        vertLines: { visible: false },
        horzLines: { visible: false },
      },
      crosshair: { mode: 0 },
      rightPriceScale: { borderColor: colors.border },
      timeScale: {
        borderColor: colors.border,
        timeVisible,
        secondsVisible,
        rightOffset,
      },
    });
    chartRef.current = chart;

    // Candlestick series with user-configurable colors
    const candleSeries = chart.addCandlestickSeries({
      upColor: up,
      downColor: down,
      borderUpColor: borderUp,
      borderDownColor: down,
      wickUpColor: borderUp,
      wickDownColor: down,
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

    // Markers
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

    // Overlay line series (indicators)
    const overlayRefs: ISeriesApi<'Line'>[] = [];
    for (const overlay of overlays) {
      const lineSeries = chart.addLineSeries({
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

    // Phase F.1: prefer the caller-supplied visibleRange when given
    // (parity comparison panes need to share an x-axis). Otherwise
    // fall back to fitContent() which auto-fits to data extent.
    if (visibleRange && visibleRange.from && visibleRange.to) {
      try {
        const fromSec = toUnixTime(visibleRange.from) as Time;
        const toSec = toUnixTime(visibleRange.to) as Time;
        if (isFinite(fromSec as number) && isFinite(toSec as number)) {
          chart.timeScale().setVisibleRange({ from: fromSec, to: toSec });
        } else {
          chart.timeScale().fitContent();
        }
      } catch {
        chart.timeScale().fitContent();
      }
    } else {
      chart.timeScale().fitContent();
    }

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
  }, [ohlcv, overlays, markers, height, getThemeColors, upColor, downColor, upBorderColor, gridLines, rightOffset, timeVisible, secondsVisible, visibleRange?.from, visibleRange?.to]);

  // M8.5: imperative live-bar update. Runs on every formingBar change; calls
  // candleSeries.update() directly — NO setData, NO re-render of this
  // component's children. This is the hot path for live chart updates
  // (called up to 4× per second).
  useEffect(() => {
    if (!formingBar || !candleSeriesRef.current) return;
    const t = toUnixTime(formingBar.time);
    if (!isFinite(t)) return;
    const bar: CandlestickData = {
      time: t as Time,
      open: Number(formingBar.open),
      high: Number(formingBar.high),
      low: Number(formingBar.low),
      close: Number(formingBar.close),
    };
    if (!isFinite(bar.open) || !isFinite(bar.high) || !isFinite(bar.low) || !isFinite(bar.close)) return;
    try {
      candleSeriesRef.current.update(bar);
    } catch {
      // LWC will throw if we try to update a time earlier than the last
      // bar — that's fine, drop silently. Next update will re-sync.
    }
  }, [formingBar]);

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
