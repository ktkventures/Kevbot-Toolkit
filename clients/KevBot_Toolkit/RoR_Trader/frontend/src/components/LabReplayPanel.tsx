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

import { useEffect, useMemo, useState, type ReactNode } from 'react';
import dynamic from 'next/dynamic';
import Card from '@/components/Card';
import ReplayControls from '@/components/ReplayControls';
import type { PaneConfig } from '@/charts/SyncedChartPane';
import { spliceAlertPanes } from '@/charts/spliceAlertLens';

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

/** Mirror of SyncedChartPane's alias map so timezone props match what the
 * chart axis labels use.  Keep in sync if either is updated. */
const _TIMEZONE_ALIAS_TO_IANA: Record<string, string> = {
  'US/Eastern': 'America/New_York',
  'US/Central': 'America/Chicago',
  'US/Mountain': 'America/Denver',
  'US/Pacific': 'America/Los_Angeles',
  'US/Alaska': 'America/Anchorage',
  'US/Hawaii': 'Pacific/Honolulu',
};

function resolveTz(tz: string | null | undefined): string {
  if (!tz) return Intl.DateTimeFormat().resolvedOptions().timeZone;
  return _TIMEZONE_ALIAS_TO_IANA[tz] || tz;
}

/** Compute the offset (seconds) of `tz` at the given UTC instant.
 * Positive = tz ahead of UTC, negative = behind. */
function tzOffsetSecs(date: Date, tz: string): number {
  const dtf = new Intl.DateTimeFormat('en-US', {
    timeZone: tz, hour12: false,
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
  const parts = dtf.formatToParts(date);
  const get = (type: string) => parseInt(parts.find(p => p.type === type)?.value ?? '0', 10);
  const asUtc = Date.UTC(get('year'), get('month') - 1, get('day'),
                         get('hour'), get('minute'), get('second'));
  return Math.round((asUtc - date.getTime()) / 1000);
}

/** Format a Unix-second timestamp as "YYYY-MM-DDTHH:MM:SS" in the user's
 * timezone — used to populate <input type="datetime-local"> fields with
 * values the user expects (matches chart axis labels). */
function unixToInputValueTz(unixSec: number, tz: string): string {
  if (!isFinite(unixSec) || unixSec <= 0) return '';
  const dtf = new Intl.DateTimeFormat('en-US', {
    timeZone: tz, hour12: false,
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
  const parts = dtf.formatToParts(new Date(unixSec * 1000));
  const get = (type: string) => parts.find(p => p.type === type)?.value ?? '00';
  return `${get('year')}-${get('month')}-${get('day')}T${get('hour')}:${get('minute')}:${get('second')}`;
}

/** Parse "YYYY-MM-DDTHH:MM:SS" interpreted as wall-clock time in `tz`,
 * return Unix seconds.  Two-pass refinement handles DST edges. */
function inputValueToUnixTz(s: string, tz: string): number {
  if (!s) return 0;
  // Step 1: parse as if UTC to get a baseline ms
  const asIfUtcMs = Date.parse(s + 'Z');
  if (!isFinite(asIfUtcMs)) return 0;
  // Step 2: get tz offset at that baseline
  let offsetSec = tzOffsetSecs(new Date(asIfUtcMs), tz);
  // Step 3: refine — offset at the candidate time may differ near DST
  let candidateMs = asIfUtcMs - offsetSec * 1000;
  offsetSec = tzOffsetSecs(new Date(candidateMs), tz);
  return Math.floor(asIfUtcMs / 1000) - offsetSec;
}

interface LabReplayPanelProps {
  algoPanes: PaneConfig[];
  /** Alert-lens panes (Lab tab path). Gate Parity passes the first/latest
   *  pair below instead and lets this component splice them per scrub head. */
  alertPanes?: PaneConfig[];
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

  // ── Gate Parity additions (all optional; Lab tab ignores them) ──
  /** When BOTH are provided, the alert lens becomes the ws_rest_spliced view:
   *  body = latest (REST), tip = first-write (WS) within grace of the scrub head. */
  alertFirstPanes?: PaneConfig[];
  alertLatestPanes?: PaneConfig[];
  /** WS-tip grace windows (seconds). Defaults to live model values. */
  gracePrimary?: number;
  graceSecondary?: number;
  /** Replay (historical splice) vs Live (real-time WS — deferred). */
  mode?: 'replay' | 'live';
  onModeChange?: (m: 'replay' | 'live') => void;
  /** Optional 1-second panes rendered under each lens. Render-functions that
   *  receive the current scrub head (Unix sec) so the 1s window can right-align. */
  oneSecBacktest?: (scrubHead: number) => ReactNode;
  oneSecAlert?: (scrubHead: number) => ReactNode;
  /** A short divergence note shown under the bar counts. */
  barCountNote?: ReactNode;
  /** A small on-screen caveat (e.g. the tip-indicator approximation). */
  tipNote?: ReactNode;

  // ── Gate Parity polish (2026-06-10) ──
  /** Hide the wordy series-title labels (e.g. "utv4 trailing stop previous")
   *  that overlay the candles, on both lenses. Numeric price labels stay. */
  hideSeriesTitles?: boolean;
  /** When provided, renders a "Labels" toggle in the header. */
  onToggleLabels?: () => void;
  /** Alert-lens WS-tip source: 'first' = decision-time (faithful default),
   *  'latest' = REST-corrected (tip heals immediately). Tip-only. */
  tipSource?: 'first' | 'latest';
  /** When provided, renders a "Tip" first/latest toggle in the header. */
  onTipSourceChange?: (s: 'first' | 'latest') => void;
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
  alertPanes = [],
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
  alertFirstPanes,
  alertLatestPanes,
  gracePrimary = 4,
  graceSecondary = 30,
  mode = 'replay',
  onModeChange,
  oneSecBacktest,
  oneSecAlert,
  barCountNote,
  tipNote,
  hideSeriesTitles = false,
  onToggleLabels,
  tipSource = 'first',
  onTipSourceChange,
}: LabReplayPanelProps) {
  // Scrub head — Unix sec. (Declared before the spliced panes that depend on it.)
  const [currentTime, setCurrentTime] = useState<number>(0);

  // Gate Parity: when first/latest panes are supplied, the alert lens is the
  // ws_rest_spliced view recomputed as the scrub head moves. Otherwise use the
  // alertPanes prop (Lab tab path).
  const spliceActive = Array.isArray(alertFirstPanes) && Array.isArray(alertLatestPanes);
  const effectiveAlertPanes = useMemo<PaneConfig[]>(() => {
    if (!spliceActive) return alertPanes;
    if (mode === 'live') return alertLatestPanes!; // Live mode deferred — show latest as a stand-in
    return spliceAlertPanes(alertFirstPanes!, alertLatestPanes!, currentTime, { gracePrimary, graceSecondary, tipSource });
  }, [spliceActive, alertPanes, alertFirstPanes, alertLatestPanes, mode, currentTime, gracePrimary, graceSecondary, tipSource]);

  // Full data extent across both lenses' candle series.
  const [fullStart, fullEnd] = useMemo(
    () => computeFullExtent(algoPanes, effectiveAlertPanes),
    [algoPanes, effectiveAlertPanes],
  );

  // User-selected window (Unix sec). Defaults to "Last 1h" of the data.
  const [windowStart, setWindowStart] = useState<number>(0);
  const [windowEnd, setWindowEnd] = useState<number>(0);
  const [windowPreset, setWindowPreset] = useState<string>('Last 1h');

  const [interval, setIntervalState] = useState<number>(defaultIntervalSec);

  // Custom start/end input buffer.  Held separately so typing in the
  // datetime-local field doesn't immediately fire-hose the chart on every
  // keystroke — only commits when the user clicks Apply (or hits Enter).
  const [customStartInput, setCustomStartInput] = useState<string>('');
  const [customEndInput, setCustomEndInput] = useState<string>('');

  // Resolve the user's preferred timezone once.  Falls back to the
  // browser's resolved timezone if the prop is null.
  const resolvedTz = useMemo(() => resolveTz(timezone), [timezone]);

  // Apply preset to window whenever data extent changes or user picks one.
  // Custom is special — it doesn't auto-derive from the extent; it uses
  // whatever the user committed via the Apply button.
  useEffect(() => {
    if (!isFinite(fullStart) || !isFinite(fullEnd) || fullEnd <= fullStart) return;
    if (windowPreset === 'Custom') {
      // Pre-fill the inputs with the current window so the user has a
      // sensible starting point to tweak.  Don't overwrite if the user
      // has already set custom values.
      if (!customStartInput) setCustomStartInput(unixToInputValueTz(windowStart || fullStart, resolvedTz));
      if (!customEndInput) setCustomEndInput(unixToInputValueTz(windowEnd || fullEnd, resolvedTz));
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
    setCustomStartInput(unixToInputValueTz(s, resolvedTz));
    setCustomEndInput(unixToInputValueTz(e, resolvedTz));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fullStart, fullEnd, windowPreset, resolvedTz]);

  const applyCustomWindow = () => {
    const s = inputValueToUnixTz(customStartInput, resolvedTz);
    const e = inputValueToUnixTz(customEndInput, resolvedTz);
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
    () => filterPanesByWindow(effectiveAlertPanes, windowStart, windowEnd),
    [effectiveAlertPanes, windowStart, windowEnd],
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
  const alertBars = useMemo(() => panesBarCount(effectiveAlertPanes), [effectiveAlertPanes]);

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
        <div className="flex items-center gap-2 text-xs flex-wrap">
          {onModeChange && (
            <div className="flex items-center gap-1">
              <span style={{ color: 'var(--text-muted)' }}>Alert lens:</span>
              <button
                onClick={() => onModeChange('replay')}
                title="Historical replay — ws_rest_spliced reconstruction (REST body + WS tip)"
                className="px-2 py-0.5 rounded transition-colors"
                style={{
                  background: mode === 'replay' ? 'var(--accent)' : 'var(--bg-input)',
                  color: mode === 'replay' ? 'white' : 'var(--text-muted)',
                  border: mode === 'replay' ? 'none' : '1px solid var(--border)',
                  cursor: 'pointer',
                }}
              >
                Replay
              </button>
              <button
                onClick={() => onModeChange('live')}
                disabled
                title="Live mode — pipes real-time WS data (coming soon)"
                className="px-2 py-0.5 rounded"
                style={{
                  background: 'var(--bg-input)',
                  color: 'var(--text-muted)',
                  border: '1px dashed var(--border)',
                  cursor: 'not-allowed',
                  opacity: 0.6,
                }}
              >
                Live (soon)
              </button>
            </div>
          )}
          {onTipSourceChange && (
            <div className="flex items-center gap-1">
              <span style={{ color: 'var(--text-muted)' }}>Tip:</span>
              {(['first', 'latest'] as const).map((s) => (
                <button
                  key={s}
                  onClick={() => onTipSourceChange(s)}
                  title={s === 'first'
                    ? 'WS tip = decision-time values (what the live engine saw) — faithful default'
                    : 'WS tip = REST-corrected (tip heals immediately; lens fully corrected)'}
                  className="px-2 py-0.5 rounded transition-colors"
                  style={{
                    background: tipSource === s ? 'var(--accent)' : 'var(--bg-input)',
                    color: tipSource === s ? 'white' : 'var(--text-muted)',
                    border: tipSource === s ? 'none' : '1px solid var(--border)',
                    cursor: 'pointer',
                  }}
                >
                  {s}
                </button>
              ))}
            </div>
          )}
          {onToggleLabels && (
            <button
              onClick={onToggleLabels}
              title={hideSeriesTitles
                ? 'Show the indicator name labels'
                : 'Hide the wordy indicator name labels that overlay the candles (price values stay)'}
              className="px-2 py-0.5 rounded transition-colors"
              style={{
                background: hideSeriesTitles ? 'var(--accent)' : 'var(--bg-input)',
                color: hideSeriesTitles ? 'white' : 'var(--text-muted)',
                border: hideSeriesTitles ? 'none' : '1px solid var(--border)',
                cursor: 'pointer',
              }}
            >
              {hideSeriesTitles ? 'Labels hidden' : 'Hide labels'}
            </button>
          )}
          <div className="flex items-center gap-1">
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
      </div>

      {barCountNote && (
        <div className="text-[11px] mb-2" style={{ color: 'var(--text-muted)' }}>
          {barCountNote}
        </div>
      )}

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
            ({resolvedTz}, second precision · press Enter or Apply to commit)
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
          {/* `key` forces a chart instance rebuild whenever the window
              commits — that triggers fitContent on the new data extent.
              Without it, LWC's setData preserves the previous visible
              range, so a window jump (e.g. Apply with a different day)
              leaves the chart showing blank space.  Currentime changes
              do NOT remount because they don't affect the key — only
              window commits do. */}
          <SyncedChartPane
            key={`algo-${windowStart}-${windowEnd}`}
            panes={algoWindowed}
            upColor={upColor}
            downColor={downColor}
            upBorderColor={upBorderColor}
            gridLines={gridLines}
            rightOffset={rightOffset}
            timezone={timezone}
            currentTime={currentTime}
            hideSeriesTitles={hideSeriesTitles}
            // Live updates explicitly suppressed in scrub mode
            formingBar={null}
            formingIndicators={null}
            formingStates={null}
            formingStateCrossTf={null}
          />
          {oneSecBacktest?.(currentTime)}
        </div>
        <div>
          <div className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>
            <strong style={{ color: 'var(--text-primary)' }}>{alertLabel}</strong>
            {alertFooter ? <span className="ml-2">{alertFooter}</span> : null}
          </div>
          <SyncedChartPane
            key={`alert-${windowStart}-${windowEnd}`}
            panes={alertWindowed}
            upColor={upColor}
            downColor={downColor}
            upBorderColor={upBorderColor}
            gridLines={gridLines}
            rightOffset={rightOffset}
            timezone={timezone}
            currentTime={currentTime}
            hideSeriesTitles={hideSeriesTitles}
            formingBar={null}
            formingIndicators={null}
            formingStates={null}
            formingStateCrossTf={null}
          />
          {oneSecAlert?.(currentTime)}
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
      {tipNote && (
        <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)', lineHeight: 1.5 }}>
          {tipNote}
        </p>
      )}
    </Card>
  );
}
