'use client';

/**
 * ChartReplayCard — M8.7 M5 (2026-05-02)
 *
 * Bar-by-bar replay for the Lab tab Alert Lens. Slim variant of
 * ScenarioReplayCard that consumes the chart-data-cache response shape
 * directly (chart_data + overlay_indicators) without scenario/trade
 * specifics. Reuses ReplayableChart + ReplayControls.
 *
 * V1 scope: price candle pane + overlay indicator lines. Trade markers
 * if provided. Skips oscillator sub-pane and heatmap (those would
 * require multi-pane scrub which is heavier — defer if we want them).
 */

import { useMemo, useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import Card from '@/components/Card';
import ReplayControls from '@/components/ReplayControls';
import type { SeriesSetup, SeriesDataInput } from '@/charts/ReplayableChart';

const ReplayableChart = dynamic(() => import('@/charts/ReplayableChart'), { ssr: false });

const INDICATOR_COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#E040FB', '#00BCD4'];

/** Price-level marker series — invisible line + cross/xcross markers
 * at exact (time, price) coordinates. Mirrors buildStrategyChartPanes'
 * algoEntry/algoExit/alertEntry/alertExit pattern so Replay matches the
 * static Chart & Trades / Lab visualization. */
export interface PriceLevelMarkerSeries {
  /** Line data points at trade prices: {time: ISO/unix, value: price} */
  data: { time: any; value: number }[];
  /** LWC marker objects keyed off the same time field */
  markers: any[];
  /** Stroke color for the (invisible) line — used for legend swatch only */
  color?: string;
  /** Legend label */
  label?: string;
}

interface ChartReplayCardProps {
  /** chart-data-cache response or equivalent */
  chartData: any[];
  overlayIndicators: string[];
  oscillatorIndicators?: string[];  // not yet rendered; reserved
  candleColorColumn?: string;
  /** Optional trade markers to draw on the candle series (entry/exit arrows) */
  candleMarkers?: any[];
  /** Price-level marker series (algo + and alert × crosses).
   *  Each becomes an invisible LWC Line series with markers attached. */
  priceLevelSeries?: PriceLevelMarkerSeries[];
  /** Theme/visual prefs (subset of chartPrefs) */
  upColor?: string;
  downColor?: string;
  gridLines?: boolean;
  rightOffset?: number;
  height?: number;
  /** Card title (e.g., "Alert Lens — Replay") */
  title?: string;
}

function toUnixSec(t: string | number | undefined): number {
  if (t == null) return 0;
  if (typeof t === 'number') return t > 1e12 ? Math.floor(t / 1000) : t;
  return Math.floor(new Date(t).getTime() / 1000);
}

export default function ChartReplayCard({
  chartData,
  overlayIndicators,
  oscillatorIndicators: _osc = [],
  candleColorColumn,
  candleMarkers = [],
  priceLevelSeries = [],
  upColor,
  downColor,
  gridLines = true,
  rightOffset = 3,
  height = 420,
  title = 'Replay',
}: ChartReplayCardProps) {
  // ID keyed on data length + overlay count + price-level series count so
  // the underlying chart re-creates (with the right number of line series)
  // when the dataset, indicator set, or marker-series count changes — e.g.
  // when cache response loads after initial render.
  const chartId = useMemo(
    () => `replay-${title.replace(/\s+/g, '-')}-${chartData.length}-${overlayIndicators.length}-${priceLevelSeries.length}`,
    [title, chartData.length, overlayIndicators.length, priceLevelSeries.length],
  );

  // Time bounds (Unix seconds) from full data set
  const { startTime, endTime, allTimes } = useMemo(() => {
    if (chartData.length === 0) {
      return { startTime: 0, endTime: 0, allTimes: [] as number[] };
    }
    const times = chartData.map(b => toUnixSec(b.timestamp ?? b.time)).filter(Boolean);
    return {
      startTime: times[0] ?? 0,
      endTime: times[times.length - 1] ?? 0,
      allTimes: times,
    };
  }, [chartData]);

  // Replay state — initialize at end (fully revealed = matches static view)
  const [currentTime, setCurrentTime] = useState<number>(endTime);
  const [interval, setInterval] = useState<number>(60);

  // Reset currentTime to endTime when dataset changes
  useEffect(() => {
    setCurrentTime(endTime);
  }, [endTime, chartData.length]);

  const isAtStart = currentTime <= startTime;
  const isAtEnd = currentTime >= endTime;

  const stepBack = () => setCurrentTime(t => Math.max(startTime, t - interval));
  const stepForward = () => setCurrentTime(t => Math.min(endTime, t + interval));
  const seek = (t: number) => setCurrentTime(Math.max(startTime, Math.min(endTime, t)));
  const reset = () => setCurrentTime(startTime);

  // Truncate chart data + overlays to currentTime
  const { truncatedCandles, truncatedOverlayData, truncatedMarkers, truncatedPriceLevels } = useMemo(() => {
    if (chartData.length === 0) {
      return {
        truncatedCandles: [], truncatedOverlayData: [], truncatedMarkers: [],
        truncatedPriceLevels: [] as PriceLevelMarkerSeries[],
      };
    }
    // Find cutoff index
    const cutoff = chartData.findIndex(
      b => toUnixSec(b.timestamp ?? b.time) > currentTime
    );
    const sliceTo = cutoff === -1 ? chartData.length : cutoff;
    const slice = chartData.slice(0, sliceTo);

    // Candle data — translate to ReplayableChart's expected shape
    const candles = slice.map((b: any) => ({
      time: toUnixSec(b.timestamp ?? b.time),
      open: b.open, high: b.high, low: b.low, close: b.close,
      ...(candleColorColumn && b[`_state_${candleColorColumn}`]
        ? { color: undefined }  // placeholder; candle coloring not yet wired
        : {}),
    }));

    // Overlay indicator lines — one series per name
    const overlayData = overlayIndicators.map((name: string) => ({
      name,
      data: slice
        .filter((b: any) => b[name] != null && isFinite(b[name]))
        .map((b: any) => ({
          time: toUnixSec(b.timestamp ?? b.time),
          value: Number(b[name]),
        })),
    }));

    // Trade markers — only show those at-or-before currentTime
    const markers = candleMarkers.filter((m: any) => toUnixSec(m.time) <= currentTime);

    // Price-level markers (algo + / alert ×) — filter both data points
    // and markers by currentTime so they reveal as the user scrubs forward.
    const priceLevels: PriceLevelMarkerSeries[] = priceLevelSeries.map(s => ({
      ...s,
      data: s.data.filter(d => toUnixSec(d.time) <= currentTime),
      markers: s.markers.filter(m => toUnixSec(m.time) <= currentTime),
    }));

    return {
      truncatedCandles: candles,
      truncatedOverlayData: overlayData,
      truncatedMarkers: markers,
      truncatedPriceLevels: priceLevels,
    };
  }, [chartData, currentTime, overlayIndicators, candleColorColumn, candleMarkers, priceLevelSeries]);

  // Build ReplayableChart props.
  // Layout: candle, then overlay lines, then 4 invisible price-level lines
  // for + (algo entry/exit) and × (alert entry/exit) cross markers.
  const seriesSetup: SeriesSetup[] = useMemo(() => {
    const setup: SeriesSetup[] = [
      { type: 'Candlestick' },
      ...overlayIndicators.map((_, i) => ({
        type: 'Line' as const,
        options: {
          color: INDICATOR_COLORS[i % INDICATOR_COLORS.length],
          lineWidth: 1.5,
        },
      })),
      ...priceLevelSeries.map(s => ({
        type: 'Line' as const,
        options: {
          color: s.color || 'rgba(255,255,255,0.2)',
          lineVisible: false,
          pointMarkersVisible: false,
          priceLineVisible: false,
          crosshairMarkerVisible: false,
          lastValueVisible: false,
          title: '',
        },
      })),
    ];
    return setup;
  }, [overlayIndicators, priceLevelSeries]);

  const seriesData: SeriesDataInput[] = useMemo(() => {
    const data: SeriesDataInput[] = [
      { data: truncatedCandles, markers: truncatedMarkers },
      ...truncatedOverlayData.map(o => ({ data: o.data })),
      ...truncatedPriceLevels.map(s => ({ data: s.data, markers: s.markers })),
    ];
    return data;
  }, [truncatedCandles, truncatedMarkers, truncatedOverlayData, truncatedPriceLevels]);

  if (chartData.length === 0) {
    return (
      <Card>
        <h4 className="text-sm font-medium mb-2">{title}</h4>
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>No bars to replay.</p>
      </Card>
    );
  }

  return (
    <Card>
      <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
        <h4 className="text-sm font-medium">
          {title}
          <span className="text-xs font-normal ml-2" style={{ color: 'var(--text-muted)' }}>
            ({truncatedCandles.length} of {chartData.length} bars revealed)
          </span>
        </h4>
      </div>

      <ReplayableChart
        id={chartId}
        height={height}
        seriesSetup={seriesSetup}
        seriesData={seriesData}
        upColor={upColor}
        downColor={downColor}
        gridLines={gridLines}
        rightOffset={rightOffset}
      />

      <div className="mt-3">
        <ReplayControls
          currentTime={currentTime}
          startTime={startTime}
          endTime={endTime}
          interval={interval}
          isAtStart={isAtStart}
          isAtEnd={isAtEnd}
          onStepBack={stepBack}
          onStepForward={stepForward}
          onSeek={seek}
          onIntervalChange={setInterval}
          onReset={reset}
        />
      </div>

      {/* Legend */}
      {overlayIndicators.length > 0 && (
        <div className="flex flex-wrap gap-3 mt-2 text-[10px]" style={{ color: 'var(--text-muted)' }}>
          {overlayIndicators.map((name: string, i: number) => (
            <span key={name} className="flex items-center gap-1">
              <span
                className="inline-block w-3 h-0.5 rounded"
                style={{ background: INDICATOR_COLORS[i % INDICATOR_COLORS.length] }}
              />
              {name.replace(/_/g, ' ')}
            </span>
          ))}
        </div>
      )}

      <p className="text-[10px] mt-2" style={{ color: 'var(--text-muted)', lineHeight: 1.5 }}>
        Step or scrub through bars to see how indicators evolved over time. Oscillator
        panes &amp; heatmap not rendered in replay mode (defer to multi-pane support).
      </p>
    </Card>
  );
}
