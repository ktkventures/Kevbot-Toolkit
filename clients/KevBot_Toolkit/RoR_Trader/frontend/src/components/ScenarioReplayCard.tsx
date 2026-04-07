'use client';

/**
 * ScenarioReplayCard — orchestrates the replay of a single scenario.
 *
 * Wires useScenarioReplay → ReplayableChart + ReplayControls + WorkflowTrace.
 * On initial load, currentTime = endTime (fully revealed, identical to static view).
 * User clicks Reset or uses controls to step through the trade.
 *
 * Data flows via props (not refs) to avoid Next.js dynamic() ref forwarding issues.
 */

import { useMemo } from 'react';
import dynamic from 'next/dynamic';
import Card from '@/components/Card';
import ReplayControls from '@/components/ReplayControls';
import WorkflowTrace from '@/components/WorkflowTrace';
import useScenarioReplay, { type ScenarioData } from '@/hooks/useScenarioReplay';
import { useChartPrefs } from '@/hooks/useChartPrefs';
import type { SeriesSetup, SeriesDataInput } from '@/charts/ReplayableChart';

const ReplayableChart = dynamic(() => import('@/charts/ReplayableChart'), { ssr: false });

// Indicator colors (match buildStrategyChartPanes)
const INDICATOR_COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#E040FB', '#00BCD4'];
const EMA_COLORS = ['#2196F3', '#FF9800', '#4CAF50'];

interface ScenarioReplayCardProps {
  scenario: any;
  displayCode: string;
}

export default function ScenarioReplayCard({ scenario, displayCode }: ScenarioReplayCardProps) {
  const replay = useScenarioReplay(scenario as ScenarioData, 300);
  const chartPrefs = useChartPrefs();

  const overlayNames = scenario.overlay_indicators || [];
  const trade = (scenario.raw_trades || [])[0];
  const isWin = (scenario.r_multiple || 0) >= 0;

  // ---- Main chart series setup (stable, set once) ----
  // Series order: [0] Candlestick, [1..N] overlay lines, [N+1] entry cross, [N+2] exit cross
  const mainSeriesSetup = useMemo((): SeriesSetup[] => {
    const setup: SeriesSetup[] = [
      { type: 'Candlestick' },
    ];
    for (let i = 0; i < overlayNames.length; i++) {
      setup.push({
        type: 'Line',
        options: {
          color: INDICATOR_COLORS[i % INDICATOR_COLORS.length],
          lineWidth: 1,
          title: overlayNames[i].replace(/_/g, ' ').toUpperCase(),
        },
      });
    }
    // Entry cross (+) — invisible line, marker only
    setup.push({
      type: 'Line',
      options: { color: '#4CAF50', lineVisible: false, pointMarkersVisible: false, priceLineVisible: false, crosshairMarkerVisible: false, lastValueVisible: false, title: '' },
    });
    // Exit cross (+) — invisible line, marker only
    setup.push({
      type: 'Line',
      options: { color: isWin ? '#4CAF50' : '#F44336', lineVisible: false, pointMarkersVisible: false, priceLineVisible: false, crosshairMarkerVisible: false, lastValueVisible: false, title: '' },
    });
    return setup;
  }, [overlayNames, isWin]);

  // ---- Main chart data (changes on every replay step) ----
  // C-type shifting is handled by the API — markers already have correct timestamps
  const mainSeriesData = useMemo((): SeriesDataInput[] => {
    const result: SeriesDataInput[] = [
      { data: replay.mainChartBars, markers: replay.mainChartMarkers },
    ];
    for (const overlay of replay.mainChartOverlays) {
      result.push({ data: overlay.data });
    }
    // Entry cross (+) — uses pre-shifted entry marker time from scenario
    const entryMarker = (scenario.markers || [])[0]; // first marker = entry
    const entryPrice = scenario.entry_price;
    if (replay.triggerStatus === 'fired' && entryMarker && entryPrice) {
      result.push({
        data: [{ time: entryMarker.time, value: entryPrice }],
        markers: [{ time: entryMarker.time, position: 'inBar', shape: 'cross', color: '#4CAF50', text: '', size: 1 }],
      });
    } else {
      result.push({ data: [] });
    }
    // Exit cross (+) — uses pre-shifted exit marker time from scenario
    const exitMarker = (scenario.markers || [])[1]; // second marker = exit
    const exitPrice = scenario.exit_price;
    if (replay.currentTime >= replay.exitTime && exitMarker && exitPrice) {
      const exitReason = trade?.exit_reason || '';
      let exitColor = isWin ? '#4CAF50' : '#F44336';
      if (exitReason === 'stop_loss') exitColor = '#FF9800';
      else if (exitReason === 'bar_count_exit') exitColor = '#26A69A';
      result.push({
        data: [{ time: exitMarker.time, value: exitPrice }],
        markers: [{ time: exitMarker.time, position: 'inBar', shape: 'cross', color: exitColor, text: '', size: 1 }],
      });
    } else {
      result.push({ data: [] });
    }
    return result;
  }, [replay.mainChartBars, replay.mainChartMarkers, replay.mainChartOverlays,
      replay.triggerStatus, replay.currentTime, replay.exitTime,
      scenario.markers, scenario.entry_price, scenario.exit_price, trade, isWin]);

  // ---- Entry Hi-Fi series setup ----
  const entrySeriesSetup = useMemo((): SeriesSetup[] => {
    const refBar = (scenario.entry_drill || [])[Math.floor((scenario.entry_drill || []).length / 2)];
    const priceLines: any[] = [];
    overlayNames.forEach((col: string, i: number) => {
      const val = refBar?.[col];
      if (val != null && isFinite(val)) {
        priceLines.push({
          price: val, color: EMA_COLORS[i % EMA_COLORS.length],
          lineWidth: 2, lineStyle: 0, axisLabelVisible: true,
          title: col.replace(/_/g, ' ').toUpperCase(),
        });
      }
    });
    if (scenario.stop_price) {
      priceLines.push({ price: scenario.stop_price, color: '#F44336', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: 'Stop' });
    }
    if (scenario.entry_price) {
      priceLines.push({ price: scenario.entry_price, color: '#4CAF50', lineWidth: 1, lineStyle: 1, axisLabelVisible: true, title: 'Entry' });
    }
    return [{ type: 'Candlestick', priceLines }];
  }, [scenario, overlayNames]);

  // ---- Entry Hi-Fi data ----
  // API already handles C-type shift — markers have correct timestamps
  const entrySeriesData = useMemo((): SeriesDataInput[] | undefined => {
    if (!replay.entryBars) return undefined;
    const markers = replay.entryFullyRevealed ? (scenario.entry_markers || []) : [];
    return [{ data: replay.entryBars, markers }];
  }, [replay.entryBars, replay.entryFullyRevealed, scenario.entry_markers]);

  // ---- Exit Hi-Fi series setup ----
  const exitSeriesSetup = useMemo((): SeriesSetup[] => {
    const refBar = (scenario.exit_drill || [])[Math.floor((scenario.exit_drill || []).length / 2)];
    const priceLines: any[] = [];
    overlayNames.forEach((col: string, i: number) => {
      const val = refBar?.[col];
      if (val != null && isFinite(val)) {
        priceLines.push({
          price: val, color: EMA_COLORS[i % EMA_COLORS.length],
          lineWidth: 2, lineStyle: 0, axisLabelVisible: true,
          title: col.replace(/_/g, ' ').toUpperCase(),
        });
      }
    });
    if (scenario.stop_price) {
      priceLines.push({ price: scenario.stop_price, color: '#F44336', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: 'Stop' });
    }
    if (scenario.exit_price) {
      priceLines.push({
        price: scenario.exit_price,
        color: isWin ? '#4CAF50' : '#F44336',
        lineWidth: 1, lineStyle: 1, axisLabelVisible: true, title: 'Exit',
      });
    }
    return [{ type: 'Candlestick', priceLines }];
  }, [scenario, overlayNames, isWin]);

  // ---- Exit Hi-Fi data ----
  // API already handles C-type shift — markers have correct timestamps
  const exitSeriesData = useMemo((): SeriesDataInput[] | undefined => {
    if (!replay.exitBars) return undefined;
    const markers = replay.exitFullyRevealed ? (scenario.exit_markers || []) : [];
    return [{ data: replay.exitBars, markers }];
  }, [replay.exitBars, replay.exitFullyRevealed, scenario.exit_markers]);

  // Format placeholder times
  const formatPlaceholderTime = (unixSec: number) => {
    const d = new Date(unixSec * 1000);
    return d.toISOString().slice(11, 19);
  };

  return (
    <Card>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <h5 className="text-sm font-medium">{scenario.name}</h5>
          {scenario.r_multiple != null && (
            <span className="text-xs font-mono font-bold"
              style={{ color: isWin ? 'var(--green)' : 'var(--red)' }}>
              {scenario.r_multiple >= 0 ? '+' : ''}{scenario.r_multiple.toFixed(1)}R
            </span>
          )}
        </div>
        <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{scenario.direction}</span>
      </div>
      <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>{scenario.description}</p>

      {/* Row 1: Main chart (60%) + Controls & Workflow (40%) */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
        {/* Main chart */}
        <div className="lg:col-span-3">
          <div style={{ minHeight: 300 }}>
            <ReplayableChart
              id={`main-${scenario.id}`}
              height={300}
              seriesSetup={mainSeriesSetup}
              seriesData={mainSeriesData}
              upColor={chartPrefs.candleUp}
              downColor={chartPrefs.candleDown}
              upBorderColor={chartPrefs.candleUpBorder}
            />
          </div>
        </div>

        {/* Controls + Workflow */}
        <div className="lg:col-span-2">
          <div className="rounded-lg p-3" style={{ background: 'var(--bg-primary)', border: '1px solid var(--border)' }}>
            {/* Replay controls */}
            <div className="mb-3">
              <ReplayControls
                currentTime={replay.currentTime}
                startTime={replay.startTime}
                endTime={replay.endTime}
                interval={replay.interval}
                isAtStart={replay.isAtStart}
                isAtEnd={replay.isAtEnd}
                onStepBack={replay.stepBackward}
                onStepForward={replay.stepForward}
                onSeek={replay.seekTo}
                onIntervalChange={replay.setInterval}
                onReset={replay.reset}
              />
            </div>

            {/* Workflow trace with monitoring + step states */}
            <h6 className="text-[10px] font-medium mb-2" style={{ color: 'var(--text-muted)' }}>Execution Workflow</h6>
            <WorkflowTrace
              steps={scenario.workflow_steps || []}
              stepStates={replay.workflowStepStates}
            />
          </div>
        </div>
      </div>

      {/* Row 2: Hi-Fi 1-second drill-downs (50/50 full width) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-3">
        {/* Entry Hi-Fi */}
        <div>
          <p className="text-[9px] font-medium mb-1" style={{ color: 'var(--text-muted)' }}>Entry Hi-Fi (1-second)</p>
          <div style={{ minHeight: 250 }}>
            {replay.entryVisible ? (
              <ReplayableChart
                id={`entry-1s-${scenario.id}`}
                height={250}
                seriesSetup={entrySeriesSetup}
                seriesData={entrySeriesData}
                upColor={chartPrefs.candleUp}
                downColor={chartPrefs.candleDown}
                upBorderColor={chartPrefs.candleUpBorder}
              />
            ) : (
              <div className="flex items-center justify-center h-full rounded-lg"
                style={{ background: 'var(--bg-primary)', border: '1px solid var(--border)', minHeight: 250 }}>
                <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                  Entry drill-down appears at {formatPlaceholderTime(replay.entryWindowStart)}
                </span>
              </div>
            )}
          </div>
        </div>
        {/* Exit Hi-Fi */}
        <div>
          <p className="text-[9px] font-medium mb-1" style={{ color: 'var(--text-muted)' }}>Exit Hi-Fi (1-second)</p>
          <div style={{ minHeight: 250 }}>
            {replay.exitVisible ? (
              <ReplayableChart
                id={`exit-1s-${scenario.id}`}
                height={250}
                seriesSetup={exitSeriesSetup}
                seriesData={exitSeriesData}
                upColor={chartPrefs.candleUp}
                downColor={chartPrefs.candleDown}
                upBorderColor={chartPrefs.candleUpBorder}
              />
            ) : (
              <div className="flex items-center justify-center h-full rounded-lg"
                style={{ background: 'var(--bg-primary)', border: '1px solid var(--border)', minHeight: 250 }}>
                <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                  Exit drill-down appears at {formatPlaceholderTime(replay.exitWindowStart)}
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
}
