'use client';

/**
 * TradeWorkflowModal — Shows the execution workflow steps for a specific trade.
 *
 * Triggered by clicking the exec badge on a trade table row.
 * Shows: mini chart (entry→exit), workflow steps with timestamps,
 * entry exec type badge, exit exec type badge.
 *
 * Reusable across: Execution Types backtest, Sandbox, Strategy Builder,
 * Strategy Detail — anywhere a trade table exists.
 */

import { useMemo } from 'react';
import Modal from '@/components/Modal';

const EXEC_BADGE_COLOR = '#2196F3';

interface TradeWorkflowModalProps {
  isOpen: boolean;
  onClose: () => void;
  trade: {
    id: number;
    entryTime: string;
    exitTime: string;
    entryPrice: number;
    exitPrice: number;
    rMultiple: number;
    execType: string;
    exitReason: string;
    direction?: string;
  };
}

function ExecBadge({ code, label }: { code: string; label?: string }) {
  return (
    <span className="text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded-full"
      style={{ color: EXEC_BADGE_COLOR, background: EXEC_BADGE_COLOR + '20' }}>
      [{code}]{label ? ` ${label}` : ''}
    </span>
  );
}

function deriveExitExecType(exitReason: string): string {
  // Stop and target exits are L-type (level cross — price hits a level)
  if (exitReason === 'stop_loss' || exitReason === 'target') return 'L';
  // Bar count exit is C-type (bar close)
  if (exitReason === 'bar_count_exit') return 'C';
  // Signal exits: derive from the exit trigger suffix
  if (exitReason.endsWith('_lc')) return 'LC';
  if (exitReason.endsWith('_cc')) return 'CC';
  if (exitReason.endsWith('_ib')) return 'L';
  // Bail exits
  if (exitReason.startsWith('unconfirmed_')) return 'C';
  // Default: C-type (bar close signal)
  return 'C';
}

function getWebhookEvent(direction: string, action: 'entry' | 'exit', orderType: string): string {
  const dir = direction.toLowerCase();
  return `${action}_${dir}_${orderType}`;
}

export default function TradeWorkflowModal({ isOpen, onClose, trade }: TradeWorkflowModalProps) {
  const direction = trade.direction || 'LONG';
  const entryExecType = trade.execType || 'C';
  const exitExecType = deriveExitExecType(trade.exitReason);
  const isWin = trade.rMultiple >= 0;

  // Build workflow steps from trade data
  const steps = useMemo(() => {
    const s: Array<{
      action: string;
      label: string;
      time?: string;
      price?: number;
      badge?: string;
      color?: string;
      isWebhook?: boolean;
    }> = [];

    // ---- ENTRY PHASE ----
    // Step 1: Trigger detection
    if (entryExecType === 'C' || entryExecType === 'CC') {
      s.push({ action: 'check_trigger', label: 'Bar-close trigger detected', time: trade.entryTime, badge: entryExecType });
    } else {
      s.push({ action: 'check_level_cross', label: 'Level cross detected within bar', time: trade.entryTime, badge: entryExecType });
    }

    // Step 2: Fill
    s.push({
      action: 'fill',
      label: `Entry filled at $${trade.entryPrice.toFixed(2)}`,
      time: trade.entryTime,
      price: trade.entryPrice,
      color: 'var(--green)',
    });

    // Step 3: Entry webhook
    s.push({
      action: 'fire_webhook',
      label: `Webhook: ${getWebhookEvent(direction, 'entry', 'market')}`,
      time: trade.entryTime,
      isWebhook: true,
    });

    // Step 4: Confirmation (LC/CC only)
    if (entryExecType === 'LC' || entryExecType === 'CC') {
      const isBailed = trade.exitReason.startsWith('unconfirmed_');
      if (isBailed) {
        s.push({
          action: 'confirmation_failed',
          label: `Confirmation FAILED — bail triggered`,
          color: 'var(--red)',
        });
      } else {
        s.push({
          action: 'confirmation_passed',
          label: `Confirmation PASSED — position confirmed`,
          color: 'var(--green)',
        });
      }
    }

    // Step 5: Position management
    s.push({
      action: 'manage_position',
      label: 'Position open — monitoring stop/target/exit triggers',
    });

    // ---- EXIT PHASE ----
    // Step 6: Exit event
    const exitLabel = trade.exitReason === 'stop_loss' ? 'Stop loss hit'
      : trade.exitReason === 'target' ? 'Target hit'
      : trade.exitReason === 'bar_count_exit' ? 'Bar count exit'
      : trade.exitReason.startsWith('unconfirmed_') ? `Bail: ${trade.exitReason.replace(/_/g, ' ')}`
      : `Exit signal: ${trade.exitReason.replace(/_/g, ' ')}`;

    s.push({
      action: 'exit',
      label: `${exitLabel} at $${trade.exitPrice.toFixed(2)} (${trade.rMultiple >= 0 ? '+' : ''}${trade.rMultiple.toFixed(2)}R)`,
      time: trade.exitTime,
      price: trade.exitPrice,
      badge: exitExecType,
      color: isWin ? 'var(--green)' : 'var(--red)',
    });

    // Step 7: Exit webhook
    s.push({
      action: 'fire_webhook',
      label: `Webhook: ${getWebhookEvent(direction, 'exit', 'market')}`,
      time: trade.exitTime,
      isWebhook: true,
    });

    return s;
  }, [trade, entryExecType, exitExecType, direction, isWin]);

  return (
    <Modal title="Trade Execution Workflow" isOpen={isOpen} onClose={onClose} width="600px">
      {/* Trade summary */}
      <div className="flex items-center justify-between mb-4 px-1">
        <div className="flex items-center gap-3">
          <span className="text-xs font-bold" style={{ color: direction === 'LONG' ? 'var(--green)' : 'var(--red)' }}>
            {direction}
          </span>
          <span className="text-xs">Trade #{trade.id}</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1">
            <span className="text-[9px]" style={{ color: 'var(--text-muted)' }}>Entry:</span>
            <ExecBadge code={entryExecType} />
          </div>
          <div className="flex items-center gap-1">
            <span className="text-[9px]" style={{ color: 'var(--text-muted)' }}>Exit:</span>
            <ExecBadge code={exitExecType} label={trade.exitReason === 'stop_loss' ? 'stop' : trade.exitReason === 'target' ? 'target' : ''} />
          </div>
          <span className="text-sm font-mono font-bold" style={{ color: isWin ? 'var(--green)' : 'var(--red)' }}>
            {trade.rMultiple >= 0 ? '+' : ''}{trade.rMultiple.toFixed(2)}R
          </span>
        </div>
      </div>

      {/* Price summary */}
      <div className="grid grid-cols-4 gap-2 mb-4">
        <div className="rounded-lg px-2 py-1.5" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
          <p className="text-[9px]" style={{ color: 'var(--text-muted)' }}>Entry</p>
          <p className="text-xs font-mono font-bold">${trade.entryPrice.toFixed(2)}</p>
        </div>
        <div className="rounded-lg px-2 py-1.5" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
          <p className="text-[9px]" style={{ color: 'var(--text-muted)' }}>Exit</p>
          <p className="text-xs font-mono font-bold">${trade.exitPrice.toFixed(2)}</p>
        </div>
        <div className="rounded-lg px-2 py-1.5" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
          <p className="text-[9px]" style={{ color: 'var(--text-muted)' }}>Entry Time</p>
          <p className="text-[10px] font-mono">{trade.entryTime}</p>
        </div>
        <div className="rounded-lg px-2 py-1.5" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
          <p className="text-[9px]" style={{ color: 'var(--text-muted)' }}>Exit Time</p>
          <p className="text-[10px] font-mono">{trade.exitTime}</p>
        </div>
      </div>

      {/* Workflow steps */}
      <div className="space-y-0">
        {steps.map((step, i) => (
          <div key={i} className="flex items-start gap-3">
            <div className="flex flex-col items-center">
              <div className="w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold flex-shrink-0"
                style={{
                  background: step.isWebhook ? 'var(--accent)' : step.action === 'exit' ? (isWin ? 'var(--green)' : 'var(--red)') : 'var(--bg-input)',
                  color: step.isWebhook || step.action === 'exit' ? 'white' : 'var(--text-muted)',
                  border: step.isWebhook || step.action === 'exit' ? 'none' : '1px solid var(--border)',
                }}>
                {i + 1}
              </div>
              {i < steps.length - 1 && (
                <div className="w-0.5 h-4" style={{ background: 'var(--border)' }} />
              )}
            </div>
            <div className="pb-2 flex-1">
              <div className="flex items-center gap-2">
                <p className="text-xs" style={{ color: step.color || (step.isWebhook ? 'var(--accent)' : 'var(--text-primary)') }}>
                  {step.label}
                </p>
                {step.badge && <ExecBadge code={step.badge} />}
              </div>
              {step.time && (
                <span className="text-[9px] font-mono" style={{ color: 'var(--text-muted)' }}>{step.time}</span>
              )}
            </div>
          </div>
        ))}
      </div>
    </Modal>
  );
}
