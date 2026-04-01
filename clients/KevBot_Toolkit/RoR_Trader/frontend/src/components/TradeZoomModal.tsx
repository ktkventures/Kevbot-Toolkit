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
}

function formatHold(seconds: number | null | undefined): string {
  if (!seconds || seconds <= 0) return '--';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

export default function TradeZoomModal({ isOpen, onClose, strategyId, tradeIdx, side, trade }: TradeZoomModalProps) {
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

    // Build markers
    const markers: any[] = [];
    const t = zoomData.trade;

    if (side === 'entry' && t.entry_time) {
      const entryTs = Math.floor(new Date(t.entry_time).getTime() / 1000);
      markers.push({
        time: entryTs,
        position: 'belowBar',
        shape: 'arrowUp',
        color: '#4CAF50',
        text: `Entry $${(t.entry_price ?? 0).toFixed(2)}`,
        size: 2,
      });
    }

    if (side === 'exit' && t.exit_time) {
      const exitTs = Math.floor(new Date(t.exit_time).getTime() / 1000);
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
    if (t.entry_price && t.entry_price > 0) {
      priceLines.push({
        price: t.entry_price,
        color: '#2196F3',
        lineWidth: 1,
        lineStyle: 1, // Dotted
        axisLabelVisible: true,
        title: 'Entry',
      });
    }

    return [{
      height: 350,
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
      }],
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
        <div style={{ height: 350 }}>
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
