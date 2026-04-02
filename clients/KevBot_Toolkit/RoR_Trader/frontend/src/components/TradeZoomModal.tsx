'use client';

/**
 * Trade Drill-Down Modal
 *
 * Shows 1-second candles around a specific trade's entry or exit bar.
 * Used to visually verify trade execution timing at sub-bar granularity.
 *
 * Features:
 * - 1-second candlestick chart via SyncedChartPane
 * - Entry/exit markers (arrows)
 * - Stop/target price level lines
 * - Trade details card
 */

import { useMemo } from 'react';
import dynamic from 'next/dynamic';
import Modal from '@/components/Modal';
import { useTradeZoom, type TradeZoomResponse } from '@/hooks/queries/useStrategies';

const SyncedChartPane = dynamic(() => import('@/charts/SyncedChartPane'), { ssr: false });

interface TradeZoomModalProps {
  isOpen: boolean;
  onClose: () => void;
  strategyId: number;
  tradeIdx: number;
  side: 'entry' | 'exit';
  trade: {
    entry_time?: string;
    exit_time?: string;
    entry_price?: number;
    exit_price?: number;
    stop_price?: number;
    target_price?: number;
    r_multiple?: number;
    exit_reason?: string;
    hold_time_seconds?: number;
    bars_held?: number;
    exec_type?: string;
  };
  /** Matching alert trade data for × marker (optional) */
  alertMatch?: {
    entryPrice?: number;
    exitPrice?: number;
    entryTime?: string;
    exitTime?: string;
  } | null;
}

function formatHold(seconds: number | null | undefined): string {
  if (!seconds || seconds <= 0) return '--';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

function _tfToSeconds(tf: string): number {
  if (tf.includes('Min')) return parseInt(tf) * 60;
  if (tf.includes('Hour') || tf === '1H') return 3600;
  if (tf.includes('Day') || tf === '1D') return 86400;
  return 60;
}

export default function TradeZoomModal({ isOpen, onClose, strategyId, tradeIdx, side, trade, alertMatch }: TradeZoomModalProps) {
  const { data: zoomData, isLoading, error } = useTradeZoom(
    isOpen ? strategyId : null,
    isOpen ? tradeIdx : null,
    side,
  );

  const chartPanes = useMemo(() => {
    if (!zoomData?.bars_1s?.length) return null;

    const bars = zoomData.bars_1s;

    // Build candlestick data
    const candleData = bars.map((b) => ({
      time: Math.floor(new Date(b.time).getTime() / 1000),
      open: b.open,
      high: b.high,
      low: b.low,
      close: b.close,
    }));

    // Build markers — apply C-type timestamp shift (plot at next bar open for candle-close triggers)
    const markers: any[] = [];
    const t = zoomData.trade;
    const tfSeconds = zoomData.timeframe ? _tfToSeconds(zoomData.timeframe) : 60;
    const isLType = (t.exec_type || 'C').includes('L');

    // Entry marker
    if (t.entry_time) {
      let entryMs = new Date(t.entry_time).getTime();
      if (!isLType) entryMs += tfSeconds * 1000; // C-type: shift to next bar open
      const entryTs = Math.floor(entryMs / 1000);
      markers.push({
        time: entryTs,
        position: 'belowBar',
        shape: 'arrowUp',
        color: '#4CAF50',
        text: `Entry $${(t.entry_price ?? 0).toFixed(2)}`,
        size: 2,
      });
    }

    // Exit marker
    if (t.exit_time) {
      const exitReason = t.exit_reason || '';
      const isLExit = ['stop_loss', 'stop', 'target'].some(r => exitReason.includes(r));
      let exitMs = new Date(t.exit_time).getTime();
      if (!isLExit) exitMs += tfSeconds * 1000; // C-type exit: shift to next bar open
      const exitTs = Math.floor(exitMs / 1000);
      const isWin = (t.r_multiple ?? 0) >= 0;
      markers.push({
        time: exitTs,
        position: 'aboveBar',
        shape: 'arrowDown',
        color: isWin ? '#4CAF50' : '#F44336',
        text: `Exit ${(t.r_multiple ?? 0) >= 0 ? '+' : ''}${(t.r_multiple ?? 0).toFixed(2)}R`,
        size: 2,
      });
    }

    // Price lines for stop/target
    const priceLines: any[] = [];
    if (t.stop_price && t.stop_price > 0) {
      priceLines.push({
        price: t.stop_price,
        color: '#F44336',
        lineWidth: 1,
        lineStyle: 2, // Dashed
        axisLabelVisible: true,
        title: 'Stop',
      });
    }
    if (t.target_price && t.target_price > 0) {
      priceLines.push({
        price: t.target_price,
        color: '#4CAF50',
        lineWidth: 1,
        lineStyle: 2,
        axisLabelVisible: true,
        title: 'Target',
      });
    }
    // Entry price shown as + point marker instead of a price line

    // Remove the entry price from priceLines (we'll show it as a + point marker instead)
    // Keep stop and target as price lines since those ARE reference levels

    // Build algo fill (+) and alert fill (×) as point marker series
    const fillMarkerSeries: any[] = [];

    // Helper: snap a timestamp to the nearest 1-second bar in the chart data
    const barTimes = candleData.map((c: any) => c.time as number);
    const snapTo1s = (ts: number): number => {
      let bestTs = barTimes[0];
      let bestDist = Infinity;
      for (const bt of barTimes) {
        const dist = Math.abs(bt - ts);
        if (dist < bestDist) { bestDist = dist; bestTs = bt; }
      }
      return bestDist < 120 ? bestTs : ts; // Within 2 minutes
    };

    // Algo fill + marker at exact entry/exit price (uses patched 'cross' shape = + symbol)
    if (t.entry_time && t.entry_price && t.entry_price > 0) {
      let algoMs = new Date(t.entry_time).getTime();
      if (!isLType) algoMs += tfSeconds * 1000;
      const algoTs = snapTo1s(Math.floor(algoMs / 1000));
      fillMarkerSeries.push({
        type: 'Line' as const,
        data: [{ time: algoTs, value: t.entry_price }],
        options: { color: '#4CAF50', lineVisible: false, pointMarkersVisible: false, priceLineVisible: false, crosshairMarkerVisible: false, lastValueVisible: false, title: '' },
        markers: [{ time: algoTs, position: 'inBar', shape: 'cross', color: '#4CAF50', text: '', size: 1 }],
      });
    }
    if (t.exit_time && t.exit_price && t.exit_price > 0) {
      const exitReason = t.exit_reason || '';
      const isLExit = ['stop_loss', 'stop', 'target'].some(r => exitReason.includes(r));
      let algoExitMs = new Date(t.exit_time).getTime();
      if (!isLExit) algoExitMs += tfSeconds * 1000;
      const algoExitTs = snapTo1s(Math.floor(algoExitMs / 1000));
      const exitColor = (t.r_multiple ?? 0) >= 0 ? '#4CAF50' : '#F44336';
      fillMarkerSeries.push({
        type: 'Line' as const,
        data: [{ time: algoExitTs, value: t.exit_price }],
        options: { color: exitColor, lineVisible: false, pointMarkersVisible: false, priceLineVisible: false, crosshairMarkerVisible: false, lastValueVisible: false, title: '' },
        markers: [{ time: algoExitTs, position: 'inBar', shape: 'cross', color: exitColor, text: '', size: 1 }],
      });
    }

    // Alert fill × marker at alert entry/exit price (uses patched 'xcross' shape = × symbol)
    if (alertMatch) {
      if (alertMatch.entryPrice && alertMatch.entryPrice > 0 && t.entry_time) {
        let alertMs = new Date(t.entry_time).getTime();
        if (!isLType) alertMs += tfSeconds * 1000;
        const alertTs = snapTo1s(Math.floor(alertMs / 1000));
        fillMarkerSeries.push({
          type: 'Line' as const,
          data: [{ time: alertTs, value: alertMatch.entryPrice }],
          options: { color: '#FF9800', lineVisible: false, pointMarkersVisible: false, priceLineVisible: false, crosshairMarkerVisible: false, lastValueVisible: false, title: '' },
          markers: [{ time: alertTs, position: 'inBar', shape: 'xcross', color: '#FF9800', text: '', size: 1 }],
        });
      }
      if (alertMatch.exitPrice && alertMatch.exitPrice > 0 && t.exit_time) {
        const exitReason = t.exit_reason || '';
        const isLExit = ['stop_loss', 'stop', 'target'].some(r => exitReason.includes(r));
        let alertExitMs = new Date(t.exit_time).getTime();
        if (!isLExit) alertExitMs += tfSeconds * 1000;
        const alertExitTs = snapTo1s(Math.floor(alertExitMs / 1000));
        fillMarkerSeries.push({
          type: 'Line' as const,
          data: [{ time: alertExitTs, value: alertMatch.exitPrice }],
          options: { color: '#FF9800', lineVisible: false, pointMarkersVisible: false, priceLineVisible: false, crosshairMarkerVisible: false, lastValueVisible: false, title: '' },
          markers: [{ time: alertExitTs, position: 'inBar', shape: 'xcross', color: '#FF9800', text: '', size: 1 }],
        });
      }
    }

    // Build stepped indicator Line series from original-timeframe data
    const indicatorSeries: any[] = [];
    const indicatorColors = ['#2196F3', '#FF9800', '#AB47BC', '#00BCD4', '#FFEB3B', '#E91E63', '#8BC34A', '#FF5722'];
    if (zoomData.indicators) {
      let colorIdx = 0;
      for (const [colName, steps] of Object.entries(zoomData.indicators as Record<string, any[]>)) {
        if (!steps || steps.length === 0) continue;
        // Skip non-overlay indicators (oscillators, volumes, etc.) — only show price-level ones
        const firstVal = steps[0]?.value ?? 0;
        const priceRange = candleData.length > 0 ? Math.max(...candleData.map(c => c.high)) : 0;
        if (firstVal <= 0 || (priceRange > 0 && (firstVal < priceRange * 0.5 || firstVal > priceRange * 1.5))) continue;

        // Build stepped line: each bar's value applies as a horizontal line until the next bar
        const lineData: any[] = [];
        for (let i = 0; i < steps.length; i++) {
          const stepTs = Math.floor(new Date(steps[i].time).getTime() / 1000);
          const nextTs = i < steps.length - 1 ? Math.floor(new Date(steps[i + 1].time).getTime() / 1000) : stepTs + 300;
          // Create a horizontal step: start of bar to start of next bar
          lineData.push({ time: stepTs, value: steps[i].value });
          if (i < steps.length - 1) {
            lineData.push({ time: nextTs - 1, value: steps[i].value });
          }
        }

        const color = indicatorColors[colorIdx % indicatorColors.length];
        indicatorSeries.push({
          type: 'Line' as const,
          data: lineData,
          options: {
            color,
            lineWidth: 1,
            lineStyle: 0, // Solid
            priceLineVisible: false,
            lastValueVisible: true,
            title: colName.replace(/_/g, ' '),
          },
        });
        colorIdx++;
      }
    }

    return [{
      height: 400,
      series: [{
        type: 'Candlestick' as const,
        data: candleData,
        options: {
          upColor: '#4CAF50',
          downColor: '#F44336',
          borderUpColor: '#4CAF50',
          borderDownColor: '#F44336',
          wickUpColor: '#4CAF50',
          wickDownColor: '#F44336',
        },
        markers,
        priceLines,
      },
      // Add stepped indicator overlay lines
      ...indicatorSeries,
      // Add algo fill (+) and alert fill (×) point markers
      ...fillMarkerSeries,
      ],
    }];
  }, [zoomData, side]);

  const t = zoomData?.trade || trade;
  const tradeNum = tradeIdx + 1;
  const sideLabel = side === 'entry' ? 'Entry' : 'Exit';

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={`Trade #${tradeNum} — ${sideLabel} Drill-Down`}
      width="900px"
    >
      {/* Trade Details Card */}
      <div className="grid grid-cols-4 sm:grid-cols-8 gap-3 mb-4">
        {[
          { label: 'Entry $', value: t.entry_price ? `$${Number(t.entry_price).toFixed(2)}` : '--' },
          { label: 'Exit $', value: t.exit_price ? `$${Number(t.exit_price).toFixed(2)}` : '--' },
          { label: 'Stop $', value: t.stop_price ? `$${Number(t.stop_price).toFixed(2)}` : '--' },
          { label: 'Target $', value: t.target_price && Number(t.target_price) > 0 ? `$${Number(t.target_price).toFixed(2)}` : '--' },
          { label: 'R', value: t.r_multiple != null ? `${Number(t.r_multiple) >= 0 ? '+' : ''}${Number(t.r_multiple).toFixed(2)}R` : '--', color: t.r_multiple != null ? (Number(t.r_multiple) >= 0 ? 'var(--green)' : 'var(--red)') : undefined },
          { label: 'Hold', value: formatHold(t.hold_time_seconds) || (t.bars_held ? `${t.bars_held} bars` : '--') },
          { label: 'Exec', value: t.exec_type || '--' },
          { label: 'Exit Reason', value: (t.exit_reason || '--').replace(/_/g, ' ') },
        ].map((item, i) => (
          <div key={i}>
            <span className="text-[10px] block" style={{ color: 'var(--text-muted)' }}>{item.label}</span>
            <span className="text-xs font-semibold" style={{ color: (item as any).color || 'var(--text-primary)' }}>
              {item.value}
            </span>
          </div>
        ))}
      </div>

      {/* 1-Second Chart */}
      {isLoading && (
        <div className="flex items-center justify-center py-12" style={{ color: 'var(--text-muted)' }}>
          <span className="text-sm">Loading 1-second bars from Polygon...</span>
        </div>
      )}

      {error && (
        <div className="flex items-center justify-center py-12" style={{ color: 'var(--red)' }}>
          <span className="text-sm">Error loading 1-second data</span>
        </div>
      )}

      {chartPanes && !isLoading && (
        <div style={{ height: 400 }}>
          <SyncedChartPane panes={chartPanes} />
        </div>
      )}

      {!isLoading && !error && (!zoomData?.bars_1s?.length) && (
        <div className="flex items-center justify-center py-12" style={{ color: 'var(--text-muted)' }}>
          <span className="text-sm">No 1-second data available for this trade</span>
        </div>
      )}

      {/* Info */}
      <p className="text-[10px] mt-3" style={{ color: 'var(--text-muted)' }}>
        Showing 1-second candles around the {sideLabel.toLowerCase()} bar.
        {zoomData?.bars_1s?.length ? ` ${zoomData.bars_1s.length} bars loaded.` : ''}
      </p>
    </Modal>
  );
}
