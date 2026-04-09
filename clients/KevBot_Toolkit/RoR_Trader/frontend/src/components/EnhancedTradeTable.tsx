'use client';

import { useState, useMemo } from 'react';
import { useDisplayStore } from '@/providers/StoreProvider';

const EXEC_BADGE_COLOR_FALLBACK = '#2196F3';

export interface TradeRow {
  id: number;
  entryTime: string;
  exitTime: string;
  direction: 'LONG' | 'SHORT';
  entryPrice: number;
  exitPrice: number;
  rMultiple: number;
  execType: string;
  exitReason: string;
  confluences: string;
}

function ExecBadge({ type }: { type: string }) {
  const execColor = useDisplayStore((s) => s.execTypeColor) || EXEC_BADGE_COLOR_FALLBACK;
  const shape = useDisplayStore((s) => s.badgeShape);
  const brackets = useDisplayStore((s) => s.showBrackets);
  return (
    <span
      className={`text-xs font-mono font-medium px-2 py-0.5 ${shape === 'square' ? 'rounded' : 'rounded-full'}`}
      style={{ color: execColor, background: execColor + '20' }}
    >
      {brackets ? `[${type}]` : type}
    </span>
  );
}

function ExitReasonBadge({ reason }: { reason: string }) {
  const colors: Record<string, string> = {
    signal: 'var(--green)',
    target: 'var(--green)',
    stop: 'var(--red)',
    bar_count: 'var(--orange)',
  };
  const c = colors[reason] || 'var(--text-muted)';
  return (
    <span
      className="text-[10px] px-1.5 py-0.5 rounded font-medium"
      style={{ color: c, background: `color-mix(in srgb, ${c} 12%, transparent)` }}
    >
      {reason}
    </span>
  );
}

interface EnhancedTradeTableProps {
  trades: TradeRow[];
  onTimeClick?: (idx: number, side: 'entry' | 'exit', trade: TradeRow) => void;
  onExecClick?: (trade: TradeRow) => void;
  onTradeNumberClick?: (trade: TradeRow) => void;
}

export default function EnhancedTradeTable({ trades, onTimeClick, onExecClick, onTradeNumberClick }: EnhancedTradeTableProps) {
  const [sortCol, setSortCol] = useState<keyof TradeRow>('id');
  const [sortAsc, setSortAsc] = useState(true);

  const sorted = useMemo(() => {
    const arr = [...trades];
    arr.sort((a, b) => {
      const av = a[sortCol];
      const bv = b[sortCol];
      if (typeof av === 'number' && typeof bv === 'number') {
        return sortAsc ? av - bv : bv - av;
      }
      return sortAsc
        ? String(av).localeCompare(String(bv))
        : String(bv).localeCompare(String(av));
    });
    return arr;
  }, [trades, sortCol, sortAsc]);

  function handleSort(col: keyof TradeRow) {
    if (sortCol === col) {
      setSortAsc(!sortAsc);
    } else {
      setSortCol(col);
      setSortAsc(true);
    }
  }

  const columns: { key: keyof TradeRow; label: string; align?: 'right' | 'center' }[] = [
    { key: 'id', label: '#', align: 'center' },
    { key: 'entryTime', label: 'Entry Time' },
    { key: 'exitTime', label: 'Exit Time' },
    { key: 'direction', label: 'Dir', align: 'center' },
    { key: 'entryPrice', label: 'Entry $', align: 'right' },
    { key: 'exitPrice', label: 'Exit $', align: 'right' },
    { key: 'rMultiple', label: 'P&L (R)', align: 'right' },
    { key: 'execType', label: 'Exec', align: 'center' },
    { key: 'exitReason', label: 'Exit Reason', align: 'center' },
  ];

  const sortIndicator = (col: keyof TradeRow) => {
    if (sortCol !== col) return '';
    return sortAsc ? ' \u2191' : ' \u2193';
  };

  return (
    <div className="overflow-x-auto" style={{ maxHeight: 400, overflowY: 'auto' }}>
      <table className="w-full text-xs">
        <thead>
          <tr style={{ borderBottom: '1px solid var(--border)' }}>
            {columns.map((col) => (
              <th
                key={col.key}
                className={`px-2 py-2 font-medium cursor-pointer hover:opacity-80 transition-opacity whitespace-nowrap sticky top-0 ${
                  col.align === 'right' ? 'text-right' : col.align === 'center' ? 'text-center' : 'text-left'
                }`}
                style={{ color: 'var(--text-muted)', background: 'var(--bg-secondary)' }}
                onClick={() => handleSort(col.key)}
              >
                {col.label}{sortIndicator(col.key)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.length === 0 ? (
            <tr><td colSpan={9} className="py-4 px-2 text-center" style={{ color: 'var(--text-muted)' }}>No trades</td></tr>
          ) : sorted.map((trade) => (
            <tr
              key={trade.id}
              className="hover:opacity-90 transition-opacity"
              style={{ borderBottom: '1px solid var(--border)' }}
            >
              {/* Trade # — clickable for replay */}
              <td
                className="px-2 py-2 text-center font-mono"
                style={{
                  color: onTradeNumberClick ? 'var(--text-secondary)' : 'var(--text-muted)',
                  cursor: onTradeNumberClick ? 'pointer' : undefined,
                  textDecoration: onTradeNumberClick ? 'underline' : undefined,
                }}
                onClick={onTradeNumberClick ? () => onTradeNumberClick(trade) : undefined}
                title={onTradeNumberClick ? 'Click to replay this trade' : undefined}
              >
                {trade.id}
              </td>

              {/* Entry Time — clickable for 1s drill-down */}
              <td
                className="px-2 py-2 whitespace-nowrap"
                style={{
                  color: 'var(--text-secondary)',
                  cursor: onTimeClick ? 'pointer' : undefined,
                  fontFamily: 'monospace', fontSize: '0.75rem',
                  textDecoration: onTimeClick ? 'underline' : undefined,
                }}
                onClick={onTimeClick ? () => onTimeClick(trade.id - 1, 'entry', trade) : undefined}
                title={onTimeClick ? 'Click to drill down into 1-second candles' : undefined}
              >
                {trade.entryTime}
              </td>

              {/* Exit Time — clickable for 1s drill-down */}
              <td
                className="px-2 py-2 whitespace-nowrap"
                style={{
                  color: 'var(--text-secondary)',
                  cursor: onTimeClick ? 'pointer' : undefined,
                  fontFamily: 'monospace', fontSize: '0.75rem',
                  textDecoration: onTimeClick ? 'underline' : undefined,
                }}
                onClick={onTimeClick ? () => onTimeClick(trade.id - 1, 'exit', trade) : undefined}
                title={onTimeClick ? 'Click to drill down into 1-second candles' : undefined}
              >
                {trade.exitTime}
              </td>

              {/* Direction */}
              <td className="px-2 py-2 text-center">
                <span className="text-[10px] font-medium" style={{ color: trade.direction === 'LONG' ? 'var(--green)' : 'var(--red)' }}>
                  {trade.direction}
                </span>
              </td>

              {/* Entry $ */}
              <td className="px-2 py-2 text-right font-mono" style={{ color: 'var(--text-secondary)' }}>
                ${trade.entryPrice.toFixed(2)}
              </td>

              {/* Exit $ */}
              <td className="px-2 py-2 text-right font-mono" style={{ color: 'var(--text-secondary)' }}>
                ${trade.exitPrice.toFixed(2)}
              </td>

              {/* P&L (R) */}
              <td className="px-2 py-2 text-right font-mono font-medium">
                <span style={{ color: trade.rMultiple >= 0 ? 'var(--green)' : 'var(--red)' }}>
                  {trade.rMultiple >= 0 ? '+' : ''}{trade.rMultiple.toFixed(2)}
                </span>
              </td>

              {/* Exec — clickable for workflow */}
              <td
                className="px-2 py-2 text-center"
                style={{ cursor: onExecClick ? 'pointer' : undefined }}
                onClick={onExecClick ? (e) => { e.stopPropagation(); onExecClick(trade); } : undefined}
                title={onExecClick ? 'Click to see execution workflow' : undefined}
              >
                <ExecBadge type={trade.execType} />
              </td>

              {/* Exit Reason */}
              <td className="px-2 py-2 text-center">
                <ExitReasonBadge reason={trade.exitReason} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
