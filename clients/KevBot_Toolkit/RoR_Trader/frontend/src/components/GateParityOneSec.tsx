'use client';

/**
 * GateParityOneSec — the 3rd layer of the Gate Parity replay: a 1-second
 * candle pane, RIGHT-aligned to the scrub head (last candle = scrub time,
 * ~`lookback` seconds before it), with the PRIMARY-TF indicator values
 * stepped onto it. For L-type triggers this shows the 1s price crossing the
 * primary-TF trigger line. Per lens (backtest = REST 1s; alert = same 1s
 * price with the alert lens's primary indicators stepped over it).
 *
 * Thin: the backend (`/window-1s`) returns only 1s bars (day-cached, fast);
 * the indicator overlays come from the lens's already-loaded primary panes.
 */
import { useMemo } from 'react';
import dynamic from 'next/dynamic';
import { useWindow1s } from '@/hooks/queries/useGateParity';
import type { PaneConfig } from '@/charts/SyncedChartPane';

const SyncedChartPane = dynamic(() => import('@/charts/SyncedChartPane'), { ssr: false });

function toUnixSec(t: any): number {
  if (t == null) return 0;
  if (typeof t === 'number') return t > 1e12 ? Math.floor(t / 1000) : t;
  return Math.floor(new Date(t).getTime() / 1000);
}

export interface OverlayLine {
  title: string;
  color: string;
  data: any[]; // primary-bar points: { time, value }
}

/** Pull the primary-TF overlay line series out of a lens's pane set. */
export function extractOverlayLines(panes: PaneConfig[]): OverlayLine[] {
  const price = (panes || []).find((p) => p.id === 'price');
  if (!price) return [];
  return (price.series || [])
    .filter((s: any) => s.liveRole === 'overlay_line' && s.type === 'Line' && Array.isArray(s.data))
    .map((s: any) => ({
      title: s.options?.title || '',
      color: s.options?.color || '#888888',
      data: s.data,
    }));
}

/** Sample the primary-TF overlay onto the EXACT 1s bar times (forward-filled:
 * each bar gets the last primary value at or before it). Aligning to the bar
 * times keeps the line on the same logical axis as the candles — otherwise
 * primary-TF times that don't match a (sparse) 1s bar add extra logical slots
 * and fitContent overshoots, squishing the pane. */
function alignToBars(points: any[], barTimes: number[]): { time: number; value: number }[] {
  const pts = (points || [])
    .map((p) => ({ t: toUnixSec(p.time ?? p.timestamp), v: Number(p.value) }))
    .filter((p) => isFinite(p.t) && isFinite(p.v))
    .sort((a, b) => a.t - b.t);
  if (!pts.length || !barTimes.length) return [];
  const out: { time: number; value: number }[] = [];
  let i = 0;
  let cur = pts[0].v;
  for (const bt of barTimes) {
    while (i < pts.length && pts[i].t <= bt) { cur = pts[i].v; i++; }
    out.push({ time: bt, value: cur });
  }
  return out;
}

interface Props {
  strategyId: number;
  scrubHead: number; // unix sec
  overlayLines: OverlayLine[];
  label: string;
  lookback?: number;
  height?: number;
  upColor?: string;
  downColor?: string;
  timezone?: string | null;
  hideSeriesTitles?: boolean;
  rightOffset?: number;
}

export default function GateParityOneSec({
  strategyId,
  scrubHead,
  overlayLines,
  label,
  lookback = 120,
  height = 240,
  upColor,
  downColor,
  timezone = null,
  hideSeriesTitles = false,
  rightOffset = 4,
}: Props) {
  // Round the scrub head to the nearest 5s so scrubbing doesn't refetch on
  // every sub-second seek (day-cached anyway, but this trims churn).
  const rounded = scrubHead > 0 ? Math.round(scrubHead / 5) * 5 : 0;
  const endIso = rounded > 0 ? new Date(rounded * 1000).toISOString() : null;
  const { data, isLoading } = useWindow1s(strategyId, endIso, lookback, rounded > 0);

  const panes = useMemo<PaneConfig[]>(() => {
    const bars = (data?.bars_1s || []).map((b) => ({
      time: toUnixSec(b.time), open: b.open, high: b.high, low: b.low, close: b.close,
    }));
    if (!bars.length) return [];
    const barTimes = bars.map((b) => b.time);
    const lineSeries = (overlayLines || []).map((ol) => ({
      type: 'Line' as const,
      data: alignToBars(ol.data, barTimes),
      options: { color: ol.color, lineWidth: 1, priceLineVisible: false, lastValueVisible: false, title: ol.title },
    }));
    return [{
      id: 'onesec',
      height,
      series: [{ type: 'Candlestick' as const, data: bars }, ...lineSeries],
    }];
  }, [data, overlayLines, height]);

  if (!endIso) return null;
  if (isLoading && !data) {
    return <div className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>1s · {label} · loading…</div>;
  }
  if (!panes.length) {
    return <div className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>1s · {label} · no 1-second bars in this window.</div>;
  }
  return (
    <div className="mt-1">
      <div className="text-[10px] mb-0.5" style={{ color: 'var(--text-muted)' }}>
        1s · {label} · last {lookback}s (ends at scrub head)
      </div>
      <SyncedChartPane
        panes={panes}
        upColor={upColor}
        downColor={downColor}
        timezone={timezone}
        rightOffset={rightOffset}
        hideSeriesTitles={hideSeriesTitles}
        currentTime={null}
        formingBar={null}
        formingIndicators={null}
        formingStates={null}
        formingStateCrossTf={null}
      />
    </div>
  );
}
