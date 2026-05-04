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
 * Intersection ensures both lenses share the same start AND end candles.
 *
 * Edge cases (covered):
 *  - One lens entirely empty → use the other's extent (still useful)
 *  - Lenses don't overlap (rare; e.g. cache first-write window doesn't
 *    intersect REST window) → fall back to the lens with the more recent
 *    data so the user sees something diagnostic instead of an error
 */
function computeFullExtent(algoPanes: PaneConfig[], alertPanes: PaneConfig[]): [number, number] {
  const [algoLo, algoHi] = panesExtent(algoPanes);
  const [alertLo, alertHi] = panesExtent(alertPanes);
  if (algoHi <= 0 && alertHi <= 0) return [0, 0];
  if (algoHi <= 0) return [alertLo, alertHi];
  if (alertHi <= 0) return [algoLo, algoHi];
  const lo = Math.max(algoLo, alertLo);
  const hi = Math.min(algoHi, alertHi);
  if (lo >= hi) {
    // No overlap — pick the lens with the more recent data so the user
    // gets a usable chart + diagnostic info instead of a blank "no bars"
    // card. The Bar-Counts header line will surface the asymmetry.
    return algoHi >= alertHi ? [algoLo, algoHi] : [alertLo, alertHi];
  }
  return [lo, hi];
}

/** Count candle bars in a panes array (for diagnostic display). */
function panesBarCount(panes: PaneConfig[]): number {
  let n = 0;
  for (const pane of panes) {
    for (const s of pane.series) {
      if (s.type === 'Candlestick') n += s.data.length;
    }
  }
  return n;
}

/** datetime-local <input> uses local-tz "YYYY-MM-DDTHH:MM:SS" — convert to/from Unix sec. */
function unixToLocalInputValue(unixSec: number): string {
  if (!isFinite(unixSec) || unixSec <= 0) return '';
  const d = new Date(unixSec * 1000);
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

function localInputValueToUnix(s: string): number {
  if (!s) return 0;
  const t = new Date(s).getTime();
  return isFinite(t) ? Math.floor(t / 1000) : 0;
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
  { label: 'Custom', seconds: -2 },      // user-provided start/end
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

  // Custom start/end input buffer.  Held separately so typing in the
  // datetime-local field doesn't immediately fire-hose the chart on every
  // keystroke — only commits when the user clicks Apply (or hits Enter).
  const [customStartInput, setCustomStartInput] = useState<string>('');
  const [customEndInput, setCustomEndInput] = useState<string>('');

  // Apply preset to window whenever data extent changes or user picks one.
  // Custom is special — it doesn't auto-derive from the extent; it uses
  // whatever the user committed via the Apply button.
  useEffect(() => {
    if (!isFinite(fullStart) || !isFinite(fullEnd) || fullEnd <= fullStart) return;
    if (windowPreset === 'Custom') {
      // Pre-fill the inputs with the current window so the user has a
      // sensible starting point to tweak.  Don't overwrite if the user
      // has already set custom values.
      if (!customStartInput) setCustomStartInput(unixToLocalInputValue(windowStart || fullStart));
      if (!customEndInput) setCustomEndInput(unixToLocalInputValue(windowEnd || fullEnd));
      return;
    }
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
    // Keep the Custom inputs in sync with the current window so switching
    // to Custom mid-flight inherits the visible range.
    setCustomStartInput(unixToLocalInputValue(s));
    setCustomEndInput(unixToLocalInputValue(e));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fullStart, fullEnd, windowPreset]);

  const applyCustomWindow = () => {
    const s = localInputValueToUnix(customStartInput);
    const e = localInputValueToUnix(customEndInput);
    if (!isFinite(s) || !isFinite(e) || s <= 0 || e <= 0 || s >= e) return;
    setWindowStart(s);
    setWindowEnd(e);
    setCurrentTime(e);
  };

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

  // Diagnostic bar counts — surfaces "alert lens has 0 bars" cases (e.g.
  // strategy 149 with first-write source on a symbol where cache hasn't
  // recorded first-write values) so the user knows WHY the chart is empty.
  const algoBars = useMemo(() => panesBarCount(algoPanes), [algoPanes]);
  const alertBars = useMemo(() => panesBarCount(alertPanes), [alertPanes]);

  if (fullEnd <= fullStart) {
    return (
      <Card>
        <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>
          No bars to compare yet.
        </p>
        <p className="text-[10px]" style={{ color: 'var(--text-muted)', lineHeight: 1.5 }}>
          Algo Lens: <strong>{algoBars}</strong> bars · Alert Lens: <strong>{alertBars}</strong> bars.
          {' '}{algoBars === 0 && alertBars === 0 && 'Both sources empty — REST + cache still loading or no data for this symbol/window.'}
          {' '}{algoBars > 0 && alertBars === 0 && 'Cache has no rows for the selected source — try toggling Latest vs First-write.'}
          {' '}{algoBars === 0 && alertBars > 0 && 'REST not loaded yet.'}
        </p>
      </Card>
    );
  }

  return (
    <Card>
      {/* Header — preset selectors + window summary + bar counts */}
      <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
        <h4 className="text-sm font-medium">
          Lab Replay
          <span className="text-xs font-normal ml-2" style={{ color: 'var(--text-muted)' }}>
            ({Math.round((windowEnd - windowStart) / 60)} min window · scrub head{' '}
            {new Date(currentTime * 1000).toISOString().slice(11, 19)} UTC ·{' '}
            Algo {algoBars} / Alert {alertBars} bars)
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

      {/* Custom datetime picker — only visible when Custom preset is active. */}
      {windowPreset === 'Custom' && (
        <div className="flex items-center gap-2 mb-3 text-xs flex-wrap" style={{ color: 'var(--text-muted)' }}>
          <span>Window:</span>
          <label className="flex items-center gap-1">
            <span>start</span>
            <input
              type="datetime-local"
              step="1"
              value={customStartInput}
              onChange={e => setCustomStartInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') applyCustomWindow(); }}
              className="px-2 py-0.5 rounded"
              style={{
                background: 'var(--bg-input)',
                border: '1px solid var(--border)',
                color: 'var(--text-primary)',
                fontFamily: 'monospace',
                fontSize: '11px',
              }}
            />
          </label>
          <label className="flex items-center gap-1">
            <span>end</span>
            <input
              type="datetime-local"
              step="1"
              value={customEndInput}
              onChange={e => setCustomEndInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') applyCustomWindow(); }}
              className="px-2 py-0.5 rounded"
              style={{
                background: 'var(--bg-input)',
                border: '1px solid var(--border)',
                color: 'var(--text-primary)',
                fontFamily: 'monospace',
                fontSize: '11px',
              }}
            />
          </label>
          <button
            onClick={applyCustomWindow}
            className="px-3 py-0.5 rounded transition-colors"
            style={{
              background: 'var(--accent)',
              color: 'white',
              border: 'none',
              cursor: 'pointer',
            }}
          >
            Apply
          </button>
          <span className="ml-2 text-[10px]" style={{ color: 'var(--text-muted)' }}>
            (browser-local time, second precision · press Enter or Apply to commit)
          </span>
        </div>
      )}

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
