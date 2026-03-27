'use client';

import { useEffect, useRef, useCallback } from 'react';
import {
  createChart, LineSeries, HistogramSeries,
  type IChartApi, type Time,
} from 'lightweight-charts';

/** Convert ISO 8601 or number to Unix seconds. */
function toUnixTime(t: string | number): number {
  if (typeof t === 'number') return t;
  return Math.floor(new Date(t).getTime() / 1000);
}

interface OscillatorData {
  /** Column name (e.g., "macd_line", "macd_signal", "macd_hist") */
  column: string;
  /** Display label */
  label: string;
  /** Hex color */
  color: string;
  /** 'line' or 'histogram' — histograms use positive/negative coloring */
  type?: 'line' | 'histogram';
}

interface OscillatorPaneProps {
  /** Full bar data — each bar has timestamp + oscillator columns */
  bars: Record<string, any>[];
  /** Which columns to render */
  series: OscillatorData[];
  height?: number;
  /** Positive histogram bar color */
  histPosColor?: string;
  /** Negative histogram bar color */
  histNegColor?: string;
}

export default function OscillatorPane({
  bars,
  series,
  height = 150,
  histPosColor = '#4CAF50',
  histNegColor = '#f44336',
}: OscillatorPaneProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  const getThemeColors = useCallback(() => {
    if (typeof window === 'undefined') return { text: '#DDD', grid: '#2B2B2B', border: '#333' };
    const s = getComputedStyle(document.documentElement);
    return {
      text: s.getPropertyValue('--text-secondary').trim() || '#DDD',
      grid: s.getPropertyValue('--border').trim() || '#2B2B2B',
      border: s.getPropertyValue('--border').trim() || '#333',
    };
  }, []);

  useEffect(() => {
    if (!containerRef.current || bars.length === 0 || series.length === 0) return;
    const colors = getThemeColors();

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height,
      layout: { background: { color: 'transparent' }, textColor: colors.text },
      grid: {
        vertLines: { color: colors.grid, style: 1 },
        horzLines: { color: colors.grid, style: 1 },
      },
      crosshair: { mode: 0 },
      rightPriceScale: { borderColor: colors.border },
      timeScale: { borderColor: colors.border, timeVisible: true, secondsVisible: false, visible: false },
    });
    chartRef.current = chart;

    // Zero reference line
    const zeroLine = chart.addSeries(LineSeries, {
      color: 'rgba(128,128,128,0.3)',
      lineWidth: 1,
      lineStyle: 2,
      priceLineVisible: false,
      lastValueVisible: false,
    });
    const zeroData = bars
      .map(b => ({ time: toUnixTime(b.timestamp) as Time, value: 0 }))
      .filter(d => isFinite(d.time as number))
      .sort((a, b) => (a.time as number) - (b.time as number));
    if (zeroData.length > 0) zeroLine.setData(zeroData);

    for (const s of series) {
      const isHist = s.type === 'histogram' || s.column.includes('hist');

      if (isHist) {
        const histSeries = chart.addSeries(HistogramSeries, {
          priceLineVisible: false,
          lastValueVisible: false,
        });
        const histData = bars
          .filter(b => b[s.column] != null)
          .map(b => ({
            time: toUnixTime(b.timestamp) as Time,
            value: Number(b[s.column]),
            color: Number(b[s.column]) >= 0 ? histPosColor : histNegColor,
          }))
          .filter(d => isFinite(d.time as number) && isFinite(d.value))
          .sort((a, b) => (a.time as number) - (b.time as number));
        if (histData.length > 0) histSeries.setData(histData);
      } else {
        const lineSeries = chart.addSeries(LineSeries, {
          color: s.color,
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
        });
        const lineData = bars
          .filter(b => b[s.column] != null)
          .map(b => ({
            time: toUnixTime(b.timestamp) as Time,
            value: Number(b[s.column]),
          }))
          .filter(d => isFinite(d.time as number) && isFinite(d.value))
          .sort((a, b) => (a.time as number) - (b.time as number));
        if (lineData.length > 0) lineSeries.setData(lineData);
      }
    }

    chart.timeScale().fitContent();

    const handleResize = () => {
      if (containerRef.current) chart.applyOptions({ width: containerRef.current.clientWidth });
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
      chartRef.current = null;
    };
  }, [bars, series, height, getThemeColors, histPosColor, histNegColor]);

  if (bars.length === 0 || series.length === 0) return null;

  return (
    <div>
      <div className="flex items-center gap-2 mb-1">
        {series.map(s => (
          <span key={s.column} className="flex items-center gap-1 text-[10px]" style={{ color: 'var(--text-muted)' }}>
            <span className="inline-block w-3 h-0.5 rounded" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
      </div>
      <div ref={containerRef} style={{ width: '100%', height }} />
    </div>
  );
}
