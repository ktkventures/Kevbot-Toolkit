'use client';

/**
 * useScenarioReplay — state machine + data slicing for scenario replay.
 *
 * Owns: currentTime, interval, derived bar slices, forming candle,
 *       workflow step states, confluence monitoring, Hi-Fi visibility.
 */

import { useState, useMemo, useCallback } from 'react';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ScenarioData {
  chart_data: any[];
  raw_trades: any[];
  entry_1s_bars: any[];
  exit_1s_bars: any[];
  workflow_steps: any[];
  overlay_indicators: string[];
  oscillator_indicators: string[];
  heatmap_conditions: any[];
  entry_price: number;
  exit_price: number;
  stop_price: number;
  target_price: number;
  r_multiple: number;
  direction: string;
  entry_markers?: any[];
  exit_markers?: any[];
  entry_drill?: any[];
  exit_drill?: any[];
}

export type StepState = 'completed' | 'active' | 'pending';

export interface ConfluenceCondition {
  label: string;
  met: boolean;
}

export interface ReplayState {
  // Time
  currentTime: number;
  startTime: number;
  endTime: number;
  totalDurationSec: number;
  elapsedSec: number;
  interval: number;
  isAtStart: boolean;
  isAtEnd: boolean;

  // Main chart (5-min bars)
  mainChartBars: any[];
  mainChartMarkers: any[];
  mainChartOverlays: { name: string; data: any[] }[];

  // Trade times (unix seconds)
  entryTime: number;
  exitTime: number;

  // Hi-Fi charts (1s bars)
  entryBars: any[] | null;
  exitBars: any[] | null;
  entryVisible: boolean;
  exitVisible: boolean;
  entryFullyRevealed: boolean;
  exitFullyRevealed: boolean;
  entryWindowStart: number;
  exitWindowStart: number;

  // Workflow
  workflowStepStates: StepState[];

  // Confluence monitoring (pre-entry)
  confluenceConditions: ConfluenceCondition[];
  confluenceAllMet: boolean;
  triggerStatus: 'waiting' | 'fired';

  // Controls
  stepForward: () => void;
  stepBackward: () => void;
  seekTo: (unixSec: number) => void;
  setInterval: (n: number) => void;
  reset: () => void;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function toUnix(ts: string | number): number {
  if (typeof ts === 'number') return ts;
  return Math.floor(new Date(ts).getTime() / 1000);
}

/**
 * Interpolate a forming candle at fraction f (0→1) through the bar.
 * At f=0 the candle is a flat line at open; at f=1 it matches the final OHLC.
 */
function interpolateFormingCandle(bar: any, f: number): any {
  const { open, high, low, close } = bar;
  const clampedF = Math.min(1, Math.max(0, f));
  return {
    time: bar.time ?? bar.timestamp,
    open,
    high: open + (high - open) * clampedF,
    low: open + (low - open) * clampedF,
    close: open + (close - open) * clampedF,
    // Semi-transparent to distinguish from closed bars
    color: 'rgba(255,255,255,0.15)',
    borderColor: 'rgba(255,255,255,0.4)',
    wickColor: 'rgba(255,255,255,0.3)',
  };
}

/** Parse a confluence record like "5M-EMA_PRICE_POSITION_V2-LMPS" into a display label. */
function parseConfluenceLabel(record: string): string {
  const parts = record.split('-');
  if (parts[0] === 'GEN') {
    // General packs: GEN-SESSION_REGULAR-IN_SESSION
    const name = parts[1]?.replace(/_/g, ' ') || record;
    const state = parts.slice(2).join('-').replace(/_/g, ' ') || '';
    return `${name}: ${state}`;
  }
  // Indicator packs: 5M-EMA_PRICE_POSITION_V2-LMPS
  const tf = parts[0];
  const interp = parts[1]?.replace(/_/g, ' ') || '';
  const state = parts.slice(2).join('-') || '';
  return `[${tf}] ${interp}: ${state}`;
}

/**
 * Evaluate simple EMA-based confluence from bar data.
 * Returns true if the bar's EMA alignment matches a LONG-favorable condition.
 */
function evaluateEmaCondition(bar: any, direction: string): boolean {
  if (!bar || bar.ema_9 == null || bar.ema_21 == null) return false;
  if (direction === 'LONG') {
    return bar.ema_9 > bar.ema_21;
  }
  return bar.ema_9 < bar.ema_21;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export default function useScenarioReplay(
  scenario: ScenarioData,
  chartBarDurationSec: number = 300,
): ReplayState {
  // Parse timestamps once
  const { startTime, endTime, entryTime, exitTime, entryWindowStart, entryWindowEnd, exitWindowStart, exitWindowEnd, barTimestamps } = useMemo(() => {
    const cd = scenario.chart_data || [];
    const st = cd.length > 0 ? toUnix(cd[0].timestamp) : 0;
    const et = cd.length > 0 ? toUnix(cd[cd.length - 1].timestamp) + chartBarDurationSec : st;

    const trade = (scenario.raw_trades || [])[0];
    const entryT = trade ? toUnix(trade.entry_time) : st;
    const exitT = trade ? toUnix(trade.exit_time) : et;

    const e1s = scenario.entry_1s_bars || [];
    const x1s = scenario.exit_1s_bars || [];
    const ews = e1s.length > 0 ? toUnix(e1s[0].timestamp) : entryT;
    const ewe = e1s.length > 0 ? toUnix(e1s[e1s.length - 1].timestamp) : entryT;
    const xws = x1s.length > 0 ? toUnix(x1s[0].timestamp) : exitT;
    const xwe = x1s.length > 0 ? toUnix(x1s[x1s.length - 1].timestamp) : exitT;

    const bts = cd.map((b: any) => toUnix(b.timestamp));

    return {
      startTime: st, endTime: et, entryTime: entryT, exitTime: exitT,
      entryWindowStart: ews, entryWindowEnd: ewe,
      exitWindowStart: xws, exitWindowEnd: xwe,
      barTimestamps: bts,
    };
  }, [scenario, chartBarDurationSec]);

  // State
  const [currentTime, setCurrentTime] = useState(endTime);
  const [interval, setIntervalState] = useState(1);

  const totalDurationSec = endTime - startTime;
  const elapsedSec = currentTime - startTime;
  const isAtStart = currentTime <= startTime;
  const isAtEnd = currentTime >= endTime;

  // Controls
  const stepForward = useCallback(() => {
    setCurrentTime(t => Math.min(t + interval, endTime));
  }, [interval, endTime]);

  const stepBackward = useCallback(() => {
    setCurrentTime(t => Math.max(t - interval, startTime));
  }, [interval, startTime]);

  const seekTo = useCallback((t: number) => {
    setCurrentTime(Math.max(startTime, Math.min(t, endTime)));
  }, [startTime, endTime]);

  const setInterval = useCallback((n: number) => { setIntervalState(n); }, []);

  const reset = useCallback(() => { setCurrentTime(startTime); }, [startTime]);

  // ---- Derived: Main chart bars ----
  const { mainChartBars, mainChartOverlays } = useMemo(() => {
    const cd = scenario.chart_data || [];
    const overlayNames = scenario.overlay_indicators || [];
    const bars: any[] = [];
    const overlayData: { name: string; data: any[] }[] = overlayNames.map(n => ({ name: n, data: [] }));

    for (let i = 0; i < cd.length; i++) {
      const barStart = barTimestamps[i];
      const barEnd = barStart + chartBarDurationSec;
      const bar = cd[i];

      if (currentTime < barStart) break; // haven't reached this bar yet

      if (currentTime >= barEnd) {
        // Completed bar — full OHLC
        bars.push({
          time: bar.timestamp,
          open: bar.open, high: bar.high, low: bar.low, close: bar.close,
        });
        // Overlay data point
        for (let oi = 0; oi < overlayNames.length; oi++) {
          const val = bar[overlayNames[oi]];
          if (val != null && isFinite(val)) {
            overlayData[oi].data.push({ time: bar.timestamp, value: val });
          }
        }
      } else {
        // Forming candle — interpolate
        const f = (currentTime - barStart) / chartBarDurationSec;
        bars.push(interpolateFormingCandle(bar, f));
        // Overlay: show interpolated value for forming bar
        for (let oi = 0; oi < overlayNames.length; oi++) {
          const val = bar[overlayNames[oi]];
          if (val != null && isFinite(val)) {
            // EMA at forming stage: interpolate from previous bar's value
            const prevBar = i > 0 ? cd[i - 1] : null;
            const prevVal = prevBar?.[overlayNames[oi]];
            if (prevVal != null && isFinite(prevVal)) {
              overlayData[oi].data.push({ time: bar.timestamp, value: prevVal + (val - prevVal) * f });
            } else {
              overlayData[oi].data.push({ time: bar.timestamp, value: val });
            }
          }
        }
      }
    }

    return { mainChartBars: bars, mainChartOverlays: overlayData };
  }, [scenario, currentTime, barTimestamps, chartBarDurationSec]);

  // ---- Derived: Main chart markers (with C-type shift to next bar open) ----
  const mainChartMarkers = useMemo(() => {
    const trade = (scenario.raw_trades || [])[0];
    if (!trade) return [];
    const markers: any[] = [];
    const dir = (scenario.direction || 'LONG').toUpperCase();
    const tfMs = chartBarDurationSec * 1000;
    const L_TYPE_EXITS = new Set(['stop_loss', 'stop', 'target', 'unconfirmed_hl']);

    // Determine if entry/exit are L-type (no shift) or C-type (shift to next bar)
    const entryTrigger = trade.entry_trigger || '';
    const execType = trade.exec_type || '';
    const entryIsLType = execType === 'L' || execType === 'LC' ||
      entryTrigger.endsWith('_ib') || entryTrigger.endsWith('_lc') ||
      entryTrigger.endsWith('_hm') || entryTrigger.endsWith('_hl');
    const exitReason = trade.exit_reason || '';
    const exitIsLType = L_TYPE_EXITS.has(exitReason);

    const shiftTime = (iso: string, isLType: boolean): string => {
      if (isLType) return iso;
      try { return new Date(new Date(iso).getTime() + tfMs).toISOString(); }
      catch { return iso; }
    };

    // Entry arrow — shifted for C-type
    if (currentTime >= entryTime) {
      markers.push({
        time: shiftTime(trade.entry_time, entryIsLType),
        position: dir === 'LONG' ? 'belowBar' : 'aboveBar',
        shape: 'arrowUp',
        color: '#4CAF50',
        text: 'Entry',
      });
    }

    // Exit arrow — shifted for C-type
    if (currentTime >= exitTime) {
      const isWin = (trade.r_multiple || 0) >= 0;
      markers.push({
        time: shiftTime(trade.exit_time, exitIsLType),
        position: dir === 'LONG' ? 'aboveBar' : 'belowBar',
        shape: 'arrowDown',
        color: isWin ? '#4CAF50' : '#F44336',
        text: exitReason === 'stop_loss' ? 'Stop' : exitReason === 'target' ? 'Target' : 'Exit',
      });
    }

    return markers;
  }, [scenario, currentTime, entryTime, exitTime, chartBarDurationSec]);

  // ---- Derived: Hi-Fi bars ----
  const { entryBars, entryVisible, entryFullyRevealed } = useMemo(() => {
    const e1s = scenario.entry_1s_bars || [];
    if (e1s.length === 0 || currentTime < entryWindowStart) {
      return { entryBars: null, entryVisible: false, entryFullyRevealed: false };
    }
    const visible = e1s.filter((b: any) => toUnix(b.timestamp) <= currentTime);
    const fullyRevealed = currentTime >= entryWindowEnd;
    return {
      entryBars: visible.map((b: any) => ({
        time: b.timestamp, open: b.open, high: b.high, low: b.low, close: b.close,
      })),
      entryVisible: true,
      entryFullyRevealed: fullyRevealed,
    };
  }, [scenario.entry_1s_bars, currentTime, entryWindowStart, entryWindowEnd]);

  const { exitBars, exitVisible, exitFullyRevealed } = useMemo(() => {
    const x1s = scenario.exit_1s_bars || [];
    if (x1s.length === 0 || currentTime < exitWindowStart) {
      return { exitBars: null, exitVisible: false, exitFullyRevealed: false };
    }
    const visible = x1s.filter((b: any) => toUnix(b.timestamp) <= currentTime);
    const fullyRevealed = currentTime >= exitWindowEnd;
    return {
      exitBars: visible.map((b: any) => ({
        time: b.timestamp, open: b.open, high: b.high, low: b.low, close: b.close,
      })),
      exitVisible: true,
      exitFullyRevealed: fullyRevealed,
    };
  }, [scenario.exit_1s_bars, currentTime, exitWindowStart, exitWindowEnd]);

  // ---- Derived: Workflow step states ----
  const workflowStepStates = useMemo((): StepState[] => {
    const steps = scenario.workflow_steps || [];
    if (steps.length === 0) return [];

    return steps.map((step: any, i: number) => {
      // Determine the effective time for this step
      let effectiveTime: number;
      if (step.time) {
        effectiveTime = toUnix(step.time);
      } else if (step.action === 'manage_position') {
        // Active between entry and exit
        if (currentTime >= exitTime) return 'completed';
        if (currentTime >= entryTime) return 'active';
        return 'pending';
      } else if (step.action === 'confirmation_passed' || step.action === 'confirmation_failed') {
        effectiveTime = entryTime;
      } else {
        // Derive from surrounding steps
        effectiveTime = i < steps.length / 2 ? entryTime : exitTime;
      }

      if (currentTime >= effectiveTime) return 'completed';

      // Is this the next step to activate?
      const prevCompleted = steps.slice(0, i).every((s: any, j: number) => {
        if (s.time) return currentTime >= toUnix(s.time);
        if (s.action === 'manage_position') return currentTime >= exitTime;
        return j < steps.length / 2 ? currentTime >= entryTime : currentTime >= exitTime;
      });
      if (prevCompleted) return 'active';

      return 'pending';
    });
  }, [scenario.workflow_steps, currentTime, entryTime, exitTime]);

  // ---- Derived: Confluence monitoring ----
  const { confluenceConditions, confluenceAllMet, triggerStatus } = useMemo(() => {
    const trade = (scenario.raw_trades || [])[0];
    const records: string[] = trade?.confluence_records || [];
    const dir = (scenario.direction || 'LONG').toUpperCase();

    // Find the current bar (last completed or forming bar)
    const cd = scenario.chart_data || [];
    let currentBar: any = null;
    for (let i = cd.length - 1; i >= 0; i--) {
      if (toUnix(cd[i].timestamp) <= currentTime) {
        currentBar = cd[i];
        break;
      }
    }

    // Build conditions from confluence_records
    const conditions: ConfluenceCondition[] = records.map(rec => {
      const label = parseConfluenceLabel(rec);
      const parts = rec.split('-');

      // For general (time-based) conditions: met if we have a current bar
      // (the fixture data is during valid trading hours)
      if (parts[0] === 'GEN') {
        return { label, met: currentBar != null };
      }

      // For EMA conditions: derive from bar data
      if (rec.includes('EMA')) {
        const met = currentBar ? evaluateEmaCondition(currentBar, dir) : false;
        return { label, met };
      }

      // Default: met when entry is reached
      return { label, met: currentTime >= entryTime };
    });

    const allMet = conditions.length > 0 && conditions.every(c => c.met);
    const triggered = currentTime >= entryTime ? 'fired' as const : 'waiting' as const;

    return { confluenceConditions: conditions, confluenceAllMet: allMet, triggerStatus: triggered };
  }, [scenario, currentTime, entryTime]);

  return {
    currentTime, startTime, endTime, totalDurationSec, elapsedSec,
    interval, isAtStart, isAtEnd,
    entryTime, exitTime,
    mainChartBars, mainChartMarkers, mainChartOverlays,
    entryBars, exitBars, entryVisible, exitVisible,
    entryFullyRevealed, exitFullyRevealed,
    entryWindowStart, exitWindowStart,
    workflowStepStates,
    confluenceConditions, confluenceAllMet, triggerStatus,
    stepForward, stepBackward, seekTo, setInterval, reset,
  };
}
