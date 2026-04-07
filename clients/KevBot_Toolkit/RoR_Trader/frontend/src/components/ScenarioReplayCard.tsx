'use client';

/**
 * ScenarioReplayCard — orchestrates the replay of a single scenario.
 *
 * Wires useScenarioReplay → ReplayableChart + ReplayControls + WorkflowTrace.
 * On initial load, currentTime = endTime (fully revealed, identical to static view).
 * User clicks Reset or uses controls to step through the trade.
 */

import { useRef, useEffect, useMemo } from 'react';
import dynamic from 'next/dynamic';
import Card from '@/components/Card';
import ReplayControls from '@/components/ReplayControls';
import WorkflowTrace from '@/components/WorkflowTrace';
import useScenarioReplay, { type ScenarioData } from '@/hooks/useScenarioReplay';
import type { ReplayableChartHandle, SeriesSetup } from '@/charts/ReplayableChart';

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
  const mainChartRef = useRef<ReplayableChartHandle>(null);
  const entryChartRef = useRef<ReplayableChartHandle>(null);
  const exitChartRef = useRef<ReplayableChartHandle>(null);

  const overlayNames = scenario.overlay_indicators || [];
  const trade = (scenario.raw_trades || [])[0];
  const isWin = (scenario.r_multiple || 0) >= 0;
  const triggerName = trade?.entry_trigger?.replace(/_/g, ' ') || 'entry signal';

  // ---- Main chart series setup (stable, set once) ----
  const mainSeriesSetup = useMemo((): SeriesSetup[] => {
    const setup: SeriesSetup[] = [
      { type: 'Candlestick' },
    ];
    // One Line series per overlay indicator
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
    return setup;
  }, [overlayNames]);

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

  // ---- Push data to charts on every replay step ----
  useEffect(() => {
    // Main chart: candlestick + overlays
    if (mainChartRef.current) {
      mainChartRef.current.setSeriesData(0, replay.mainChartBars);
      mainChartRef.current.setSeriesMarkers(0, replay.mainChartMarkers);
      // Overlay line series (series index 1, 2, 3, ...)
      for (let i = 0; i < replay.mainChartOverlays.length; i++) {
        mainChartRef.current.setSeriesData(i + 1, replay.mainChartOverlays[i].data);
      }
      mainChartRef.current.fitContent();
    }

    // Entry Hi-Fi
    if (entryChartRef.current && replay.entryBars) {
      entryChartRef.current.setSeriesData(0, replay.entryBars);
      if (replay.entryFullyRevealed && scenario.entry_markers) {
        entryChartRef.current.setSeriesMarkers(0, scenario.entry_markers);
      }
      entryChartRef.current.fitContent();
    }

    // Exit Hi-Fi
    if (exitChartRef.current && replay.exitBars) {
      exitChartRef.current.setSeriesData(0, replay.exitBars);
      if (replay.exitFullyRevealed && scenario.exit_markers) {
        exitChartRef.current.setSeriesMarkers(0, scenario.exit_markers);
      }
      exitChartRef.current.fitContent();
    }
  }, [
    replay.currentTime, replay.mainChartBars, replay.mainChartMarkers,
    replay.mainChartOverlays, replay.entryBars, replay.exitBars,
    replay.entryFullyRevealed, replay.exitFullyRevealed,
    scenario.entry_markers, scenario.exit_markers,
  ]);

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
              ref={mainChartRef}
              id={`main-${scenario.id}`}
              height={300}
              seriesSetup={mainSeriesSetup}
            />
          </div>
        </div>

        {/* Controls + Workflow */}
        <div className="lg:col-span-2">
          <div className="rounded-lg p-3" style={{ background: 'var(--bg-primary)', border: '1px solid var(--border)' }}>
            {/* Replay controls — in the gap above workflow */}
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
              confluenceConditions={replay.confluenceConditions}
              confluenceAllMet={replay.confluenceAllMet}
              triggerStatus={replay.triggerStatus}
              triggerName={triggerName}
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
                ref={entryChartRef}
                id={`entry-1s-${scenario.id}`}
                height={250}
                seriesSetup={entrySeriesSetup}
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
                ref={exitChartRef}
                id={`exit-1s-${scenario.id}`}
                height={250}
                seriesSetup={exitSeriesSetup}
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
