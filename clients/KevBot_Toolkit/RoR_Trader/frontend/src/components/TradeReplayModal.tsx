'use client';

import Modal from '@/components/Modal';
import dynamic from 'next/dynamic';

const ScenarioReplayCard = dynamic(() => import('@/components/ScenarioReplayCard'), { ssr: false });

interface TradeReplayModalProps {
  isOpen: boolean;
  onClose: () => void;
  tradeNum: number;
  execType: string;
  direction: string;
  rMultiple: number;
  entryPrice: number;
  exitPrice: number;
  scenarioData: any | null;
  isLoading: boolean;
  error?: string | null;
}

export default function TradeReplayModal({
  isOpen,
  onClose,
  tradeNum,
  execType,
  direction,
  rMultiple,
  entryPrice,
  exitPrice,
  scenarioData,
  isLoading,
  error,
}: TradeReplayModalProps) {
  return (
    <Modal
      title={`Trade #${tradeNum} Replay`}
      isOpen={isOpen}
      onClose={onClose}
      width="95vw"
    >
      {/* Trade summary bar */}
      <div
        className="flex items-center gap-4 mb-4 text-xs font-mono px-1 py-2 rounded"
        style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}
      >
        <span style={{ color: 'var(--text-muted)' }}>Trade #{tradeNum}</span>
        <span style={{ color: direction === 'LONG' ? 'var(--green)' : 'var(--red)' }}>{direction}</span>
        <span style={{ color: 'var(--text-secondary)' }}>Entry ${entryPrice.toFixed(2)}</span>
        <span style={{ color: 'var(--text-secondary)' }}>Exit ${exitPrice.toFixed(2)}</span>
        <span style={{ color: rMultiple >= 0 ? 'var(--green)' : 'var(--red)' }}>
          {rMultiple >= 0 ? '+' : ''}{rMultiple.toFixed(2)}R
        </span>
        <span
          className="text-xs font-mono font-medium px-2 py-0.5 rounded-full"
          style={{ color: '#2196F3', background: '#2196F320' }}
        >
          {execType}
        </span>
      </div>

      {/* Content */}
      {isLoading && (
        <div className="flex items-center justify-center py-16">
          <div className="text-sm" style={{ color: 'var(--text-muted)' }}>
            Loading replay data...
          </div>
        </div>
      )}

      {error && !isLoading && (
        <div className="flex items-center justify-center py-16">
          <div className="text-sm" style={{ color: 'var(--red)' }}>
            {error}
          </div>
        </div>
      )}

      {scenarioData && !isLoading && !error && (
        <ScenarioReplayCard scenario={scenarioData} displayCode={execType} />
      )}
    </Modal>
  );
}
