'use client';

/**
 * LabReplayPanel — M8.7 M5 (2026-05-04) unified Algo|Alert side-by-side
 * with shared replay scrub.
 *
 * Replaces the V1 ChartReplayCard path.  Each lens uses the same
 * SyncedChartPane renderer that the static charts use, in scrub mode.
 * That means indicators, oscillators, heatmap, and trade markers all
 * render identically across the two lenses — divergence between Algo
 * (REST) and Alert (live cache) is the actual signal, not a rendering
 * artifact.
 *
 *   ┌──────────────── Window: [start ◀] [end ▶] [Last 1h] [All] ───┐
 *   │ Algo Lens (REST)              │ Alert Lens (live cache)      │
 *   │ ┌──────────────────────────┐  │ ┌──────────────────────────┐ │
 *   │ │ candles + indicators +   │  │ │ candles + indicators +   │ │
 *   │ │ oscillator + heatmap     │  │ │ oscillator + heatmap     │ │
 *   │ └──────────────────────────┘  │ └──────────────────────────┘ │
 *   ├───────────────────────────────────────────────────────────────┤
 *   │ ◀ ▶  HH:MM:SS  step: [1s ▾]   ─────────░░░░░ scrub bar       │
 *   └───────────────────────────────────────────────────────────────┘
 */

import { useEffect, useMemo, useState } from 'react';
import dynamic from 'next/dynamic';
import Card from '@/components/Card';
import ReplayControls from '@/components/ReplayControls';
import type { PaneConfig } from '@/charts/SyncedChartPane';

const SyncedChartPane = dynamic(() => import('@/charts/SyncedChartPane'), { ssr: false });

function toUnixSec(t: any): number {
  if (t == null) return 0;
  if (typeof t === 'number') return t > 1e12 ? Math.floor(t / 1000) : t;
  return Math.floor(new Date(t).getTime() / 1000);
}

/** Filter a panes array's series data and markers to [start, end] (Unix sec). */
function filterPanesByWindow(panes: PaneConfig[], start: number, end: number): PaneConfig[] {
  if (!isFinite(start) || !isFinite(end) || start >= end) return panes;
  return panes.map(pane => ({
    ...pane,
    series: pane.series.map(s => ({
      ...s,
      data: s.data.filter((d: any) => {
        const t = toUnixSec(d.time ?? d.timestamp);
        return t >= start && t <= end;
      }),
      markers: (s.markers || []).filter((m: any) => {
        const t = toUnixSec(m.time);
        return t >= start && t <= end;
      }),
    })),
  }));
}

/** Compute the candle time extent of a single panes array. */
function panesExtent(panes: PaneConfig[]): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const pane of panes) {
    for (const s of pane.series) {
      if (s.type !== 'Candlestick') continue;
      for (const d of s.data) {
        const t = toUnixSec(d.time ?? d.timestamp);
        if (!isFinite(t)) continue;
        if (t < lo) lo = t;
        if (t > hi) hi = t;
      }
    }
  }
  return [isFinite(lo) ? lo : 0, isFinite(hi) ? hi : 0];
}

/** Apples-to-apples extent: only show the time range where BOTH lenses have
 * data. Algo Lens (REST) typically settles ~15 min behind WS, so naive max
 * picks the Alert end and leaves the Algo chart stretched/empty on the right.
 * Intersection ensures both lenses share the same start AND end candles. */
function computeFullExtent(algoPanes: PaneConfig[], alertPanes: PaneConfig[]): [number, number] {
  const [algoLo, algoHi] = panesExtent(algoPanes);
  const [alertLo, alertHi] = panesExtent(alertPanes);
  // If a lens is empty (extent = [0,0]), fall through to the other lens's extent.
  if (algoHi <= 0) return [alertLo, alertHi];
  if (alertHi <= 0) return [algoLo, algoHi];
  return [Math.max(algoLo, alertLo), Math.min(algoHi, alertHi)];
}

interface LabReplayPanelProps {
  algoPanes: PaneConfig[];
  alertPanes: PaneConfig[];
  algoLabel?: string;
  alertLabel?: string;
  algoFooter?: string;
  alertFooter?: string;
  /** Display prefs */
  upColor?: string;
  downColor?: string;
  upBorderColor?: string;
  gridLines?: boolean;
  rightOffset?: number;
  timezone?: string | null;
  /** Per-lens chart height */
  height?: number;
  /** Default scrub step in seconds (typically the strategy's primary TF). */
  defaultIntervalSec?: number;
}

const PRESET_WINDOWS = [
  { label: 'Last 1h', seconds: 3600 },
  { label: 'Last 4h', seconds: 4 * 3600 },
  { label: 'Today', seconds: 0 },        // special — handled below
  { label: 'All', seconds: -1 },
] as const;

export default function LabReplayPanel({
  algoPanes,
  alertPanes,
  algoLabel = 'Algo Lens',
  alertLabel = 'Alert Lens',
  algoFooter,
  alertFooter,
  upColor,
  downColor,
  upBorderColor,
  gridLines = true,
  rightOffset = 3,
  timezone = null,
  height = 350,
  defaultIntervalSec = 60,
}: LabReplayPanelProps) {
  // Full data extent across both lenses' candle series.
  const [fullStart, fullEnd] = useMemo(
    () => computeFullExtent(algoPanes, alertPanes),
    [algoPanes, alertPanes],
  );

  // User-selected window (Unix sec). Defaults to "Last 1h" of the data.
  const [windowStart, setWindowStart] = useState<number>(0);
  const [windowEnd, setWindowEnd] = useState<number>(0);
  const [windowPreset, setWindowPreset] = useState<string>('Last 1h');

  // Scrub head — Unix sec.
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [interval, setIntervalState] = useState<number>(defaultIntervalSec);

  // Apply preset to window whenever data extent changes or user picks one.
  useEffect(() => {
    if (!isFinite(fullStart) || !isFinite(fullEnd) || fullEnd <= fullStart) return;
    let s = fullStart;
    let e = fullEnd;
    if (windowPreset === 'Last 1h') s = Math.max(fullStart, fullEnd - 3600);
    else if (windowPreset === 'Last 4h') s = Math.max(fullStart, fullEnd - 4 * 3600);
    else if (windowPreset === 'Today') {
      const d = new Date(fullEnd * 1000);
      d.setUTCHours(13, 30, 0, 0);  // 09:30 ET ≈ 13:30 UTC during EDT
      s = Math.max(fullStart, Math.floor(d.getTime() / 1000));
    } else if (windowPreset === 'All') s = fullStart;
    setWindowStart(s);
    setWindowEnd(e);
    setCurrentTime(e); // start fully revealed
  }, [fullStart, fullEnd, windowPreset]);

  // Filter both panes to the active window so each lens' chart only
  // contains points the user wants to inspect.  Scrub-mode within
  // SyncedChartPane further slices to ≤ currentTime.
  const algoWindowed = useMemo(
    () => filterPanesByWindow(algoPanes, windowStart, windowEnd),
    [algoPanes, windowStart, windowEnd],
  );
  const alertWindowed = useMemo(
    () => filterPanesByWindow(alertPanes, windowStart, windowEnd),
    [alertPanes, windowStart, windowEnd],
  );

  const isAtStart = currentTime <= windowStart;
  const isAtEnd = currentTime >= windowEnd;

  const stepBack = () => setCurrentTime(t => Math.max(windowStart, t - interval));
  const stepForward = () => setCurrentTime(t => Math.min(windowEnd, t + interval));
  const seek = (t: number) => setCurrentTime(Math.max(windowStart, Math.min(windowEnd, t)));
  const reset = () => setCurrentTime(windowStart);

  if (fullEnd <= fullStart) {
    return (
      <Card>
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          No bars to compare yet.
        </p>
      </Card>
    );
  }

  return (
    <Card>
      {/* Header — preset selectors + window summary */}
      <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
        <h4 className="text-sm font-medium">
          Lab Replay
          <span className="text-xs font-normal ml-2" style={{ color: 'var(--text-muted)' }}>
            ({Math.round((windowEnd - windowStart) / 60)} min window · scrub head{' '}
            {new Date(currentTime * 1000).toISOString().slice(11, 19)} UTC)
          </span>
        </h4>
        <div className="flex items-center gap-1 text-xs">
          {PRESET_WINDOWS.map(p => (
            <button
              key={p.label}
              onClick={() => setWindowPreset(p.label)}
              className="px-2 py-0.5 rounded transition-colors"
              style={{
                background: windowPreset === p.label ? 'var(--accent)' : 'var(--bg-input)',
                color: windowPreset === p.label ? 'white' : 'var(--text-muted)',
                border: windowPreset === p.label ? 'none' : '1px solid var(--border)',
                cursor: 'pointer',
              }}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Side-by-side lenses */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div>
          <div className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>
            <strong style={{ color: 'var(--text-primary)' }}>{algoLabel}</strong>
            {algoFooter ? <span className="ml-2">{algoFooter}</span> : null}
          </div>
          <SyncedChartPane
            panes={algoWindowed}
            upColor={upColor}
            downColor={downColor}
            upBorderColor={upBorderColor}
            gridLines={gridLines}
            rightOffset={rightOffset}
            timezone={timezone}
            currentTime={currentTime}
            // Live updates explicitly suppressed in scrub mode
            formingBar={null}
            formingIndicators={null}
            formingStates={null}
            formingStateCrossTf={null}
          />
        </div>
        <div>
          <div className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>
            <strong style={{ color: 'var(--text-primary)' }}>{alertLabel}</strong>
            {alertFooter ? <span className="ml-2">{alertFooter}</span> : null}
          </div>
          <SyncedChartPane
            panes={alertWindowed}
            upColor={upColor}
            downColor={downColor}
            upBorderColor={upBorderColor}
            gridLines={gridLines}
            rightOffset={rightOffset}
            timezone={timezone}
            currentTime={currentTime}
            formingBar={null}
            formingIndicators={null}
            formingStates={null}
            formingStateCrossTf={null}
          />
        </div>
      </div>

      {/* Shared replay controls */}
      <div className="mt-3">
        <ReplayControls
          currentTime={currentTime}
          startTime={windowStart}
          endTime={windowEnd}
          interval={interval}
          isAtStart={isAtStart}
          isAtEnd={isAtEnd}
          onStepBack={stepBack}
          onStepForward={stepForward}
          onSeek={seek}
          onIntervalChange={setIntervalState}
          onReset={reset}
        />
      </div>

      <p className="text-[10px] mt-2" style={{ color: 'var(--text-muted)', lineHeight: 1.5 }}>
        Both lenses share the scrub head — divergence between Algo (REST) and
        Alert (live cache) at any moment is visually obvious.  Indicators,
        oscillators, heatmap and trade markers render identically because both
        lenses use the same renderer.
      </p>
    </Card>
  );
}
