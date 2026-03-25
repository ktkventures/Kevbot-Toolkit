'use client';

import { useState, useMemo, useEffect } from 'react';
import Link from 'next/link';
import Card from '@/components/Card';
import Modal from '@/components/Modal';

/* ========================================================================= */
/* MOCK DATA                                                                  */
/* ========================================================================= */

interface Strategy {
  id: string;
  name: string;
  symbol: string;
  direction: 'LONG' | 'SHORT';
  timeframe: string;
  session: string;
  status: 'On Track' | 'Outperforming' | 'Underperforming' | 'Insufficient Data';
  tags: string[];
  winRate: number;
  pf: number;
  dailyR: number;
  dailyROI: number;
  trades: number;
  maxDD: number;
  fwdWinRate: number | null;
  fwdPF: number | null;
  fwdDailyROI: number | null;
  fwdTrades: number;
  fwdSince: string;
  alertWinRate: number | null;
  alertPF: number | null;
  alertDailyR: number | null;
  alertDailyROI: number | null;
  alertTrades: number;
  alertMaxDD: number | null;
  entry: string;
  exit: string[];
  stop: string;
  target: string;
  confluence: string[];
  alertTracking: boolean;
  monitored: boolean;
  btDays: number;
}

const mockStrategies: Strategy[] = [
  {
    id: '1', name: 'NVDA LONG - Mass #2', symbol: 'NVDA', direction: 'LONG', timeframe: '1Min',
    session: 'RTH', status: 'On Track', tags: ['Scalping', 'High PF'],
    winRate: 54.0, pf: 2.05, dailyR: 1.95, dailyROI: 0.42, trades: 224, maxDD: -2.5,
    fwdWinRate: 52.1, fwdPF: 1.82, fwdDailyROI: 0.38, fwdTrades: 18, fwdSince: '2026-03-07',
    alertWinRate: 50.5, alertPF: 1.70, alertDailyR: 1.75, alertDailyROI: 0.35, alertTrades: 16, alertMaxDD: -2.8,
    entry: '[C] EMA Stack (Default) > Short > Mid Cross',
    exit: ['[C] EMA Stack (Default) > Short < Mid Cross'],
    stop: 'Swing Stop (Default)', target: 'R:R (Default)',
    confluence: ['5M-RVOL-HIGH', '1D-MACD_LINE-BULL'], alertTracking: true, monitored: true, btDays: 90,
  },
  {
    id: '2', name: 'SPY LONG - Mass #1', symbol: 'SPY', direction: 'LONG', timeframe: '5Min',
    session: 'RTH', status: 'Outperforming', tags: ['Swing'],
    winRate: 62.5, pf: 3.12, dailyR: 2.41, dailyROI: 0.18, trades: 89, maxDD: -1.8,
    fwdWinRate: 65.0, fwdPF: 3.45, fwdDailyROI: 0.20, fwdTrades: 12, fwdSince: '2026-03-01',
    alertWinRate: 63.2, alertPF: 3.20, alertDailyR: 2.30, alertDailyROI: 0.19, alertTrades: 11, alertMaxDD: -1.9,
    entry: '[L] VWAP (Default) > Cross Above VWAP',
    exit: ['[C] VWAP (Default) > Cross Below VWAP'],
    stop: 'Swing Stop (Wide)', target: 'R:R (3:1 Aggressive)',
    confluence: ['1H-EMA_STACK-SML'], alertTracking: true, monitored: true, btDays: 120,
  },
  {
    id: '3', name: 'AAPL LONG - Mass #5', symbol: 'AAPL', direction: 'LONG', timeframe: '1Min',
    session: 'RTH', status: 'Insufficient Data', tags: ['New'],
    winRate: 48.2, pf: 1.45, dailyR: 0.82, dailyROI: 0.15, trades: 156, maxDD: -3.1,
    fwdWinRate: null, fwdPF: null, fwdDailyROI: null, fwdTrades: 3, fwdSince: '2026-03-18',
    alertWinRate: null, alertPF: null, alertDailyR: null, alertDailyROI: null, alertTrades: 0, alertMaxDD: null,
    entry: '[C] MACD Line (Default) > Bullish Cross',
    exit: ['[C] MACD Line (Default) > Bearish Cross', '[C] Bar Count Exit (Default) > Bar Count Exit'],
    stop: 'ATR Stop (Default)', target: 'R:R (Default)',
    confluence: ['5M-RVOL-HIGH', '1D-EMA_STACK-SML'], alertTracking: false, monitored: false, btDays: 60,
  },
  {
    id: '4', name: 'TSLA LONG - Mass #5', symbol: 'TSLA', direction: 'LONG', timeframe: '1Min',
    session: 'RTH', status: 'On Track', tags: ['Scalping'],
    winRate: 51.3, pf: 1.78, dailyR: 1.23, dailyROI: 0.31, trades: 201, maxDD: -2.9,
    fwdWinRate: 50.0, fwdPF: 1.65, fwdDailyROI: 0.28, fwdTrades: 8, fwdSince: '2026-03-10',
    alertWinRate: 48.5, alertPF: 1.55, alertDailyR: 1.10, alertDailyROI: 0.26, alertTrades: 7, alertMaxDD: -3.1,
    entry: '[LC] UT Bot (Default) > Buy Signal',
    exit: ['[LC] UT Bot (Default) > Sell Signal'],
    stop: 'Swing Stop (Default)', target: 'R:R (Default)',
    confluence: [], alertTracking: true, monitored: false, btDays: 90,
  },
  {
    id: '5', name: 'META LONG - Mass #13', symbol: 'META', direction: 'LONG', timeframe: '5Min',
    session: 'RTH', status: 'Underperforming', tags: ['Swing', 'Low DD'],
    winRate: 45.8, pf: 1.12, dailyR: 0.34, dailyROI: 0.08, trades: 67, maxDD: -4.2,
    fwdWinRate: 40.0, fwdPF: 0.95, fwdDailyROI: 0.05, fwdTrades: 15, fwdSince: '2026-03-05',
    alertWinRate: null, alertPF: null, alertDailyR: null, alertDailyROI: null, alertTrades: 0, alertMaxDD: null,
    entry: '[C] EMA Stack (Scalping) > Short > Mid Cross',
    exit: ['[C] EMA Stack (Scalping) > Short < Mid Cross'],
    stop: 'Swing Stop (Wide)', target: 'R:R (Default)',
    confluence: ['5M-RVOL-HIGH'], alertTracking: false, monitored: false, btDays: 150,
  },
];

/* ========================================================================= */
/* HELPERS                                                                     */
/* ========================================================================= */

const statusColors: Record<string, string> = {
  'On Track': 'var(--green)',
  'Outperforming': 'var(--blue)',
  'Underperforming': 'var(--red)',
  'Insufficient Data': 'var(--text-muted)',
};

function daysSince(dateStr: string): number {
  const d = new Date(dateStr);
  const now = new Date();
  return Math.floor((now.getTime() - d.getTime()) / 86400000);
}

/** Generate a mini equity curve SVG from mock cumulative R data */
// Display settings V5 equity curve colors
const EXEC_BADGE_COLOR = '#2196F3';
const EQ_BT_COLOR = '#2196F3';
const EQ_FWD_COLOR = '#FF9800';
const EQ_LIVE_COLOR = '#4CAF50';

function MiniEquityCurve({ strategyId, fwdStartPct, hasAlerts, showHWM, showEdgeMA, showConfBands, height = 64 }: { strategyId: string; fwdStartPct: number; hasAlerts: boolean; showHWM: boolean; showEdgeMA: boolean; showConfBands: boolean; height?: number }) {
  const seed = parseInt(strategyId, 10) || 1;
  const totalPoints = 50;
  const fwdIdx = Math.max(1, Math.floor(totalPoints * fwdStartPct));
  const w = 320;
  const h = height;
  const pad = 3;

  // Generate backtest equity curve
  const btPoints: number[] = [0];
  let val = 0;
  for (let i = 1; i <= totalPoints; i++) {
    val += Math.sin(seed * 137.5 * i) * 0.5 + 0.2;
    btPoints.push(val);
  }

  // Forward test continues from backtest — same trajectory but diverges slightly
  const fwdPoints = btPoints.slice(fwdIdx).map((p, i) => p + Math.sin(seed * 42 * (fwdIdx + i)) * 0.3);

  // Live alerts overlay on forward test x-range — shows slippage (slightly below/above fwd)
  const livePoints = hasAlerts ? fwdPoints.map((p, i) => p + Math.sin(seed * 99 * i) * 0.4 - 0.3) : [];

  const allVals = [...btPoints, ...fwdPoints, ...livePoints];
  const min = Math.min(...allVals);
  const max = Math.max(...allVals);
  const range = max - min || 1;

  const toY = (v: number) => h - ((v - min) / range) * (h - pad * 2) - pad;
  const toX = (idx: number) => (idx / totalPoints) * w;

  const buildLine = (pts: number[], startIdx: number) =>
    pts.map((p, i) => `${toX(startIdx + i).toFixed(1)},${toY(p).toFixed(1)}`).join(' ');
  const buildFill = (pts: number[], startIdx: number) => {
    const line = buildLine(pts, startIdx);
    return `${toX(startIdx).toFixed(1)},${h} ${line} ${toX(startIdx + pts.length - 1).toFixed(1)},${h}`;
  };

  // HWM
  let hwmPeak = -Infinity;
  const hwmLine = btPoints.map((p, i) => { hwmPeak = Math.max(hwmPeak, p); return `${toX(i).toFixed(1)},${toY(hwmPeak).toFixed(1)}`; }).join(' ');

  const zeroY = toY(0);
  const bndX = toX(fwdIdx);

  return (
    <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" style={{ display: 'block' }}>
      <defs>
        <linearGradient id={`btG${seed}`} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={EQ_BT_COLOR} stopOpacity="0.2" /><stop offset="100%" stopColor={EQ_BT_COLOR} stopOpacity="0" /></linearGradient>
        <linearGradient id={`fwG${seed}`} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={EQ_FWD_COLOR} stopOpacity="0.15" /><stop offset="100%" stopColor={EQ_FWD_COLOR} stopOpacity="0" /></linearGradient>
      </defs>
      {/* Zero line */}
      <line x1="0" y1={zeroY} x2={w} y2={zeroY} stroke="var(--text-muted)" strokeWidth="0.5" strokeDasharray="4 3" opacity="0.4" />
      {/* HWM */}
      {showHWM && <polyline points={hwmLine} fill="none" stroke="var(--green)" strokeWidth="0.5" strokeDasharray="2 2" opacity="0.4" />}
      {/* Edge MA (simple moving average of equity) */}
      {showEdgeMA && (() => {
        const maLine = btPoints.map((_, i) => {
          const window = btPoints.slice(Math.max(0, i - 7), i + 1);
          const avg = window.reduce((s, v) => s + v, 0) / window.length;
          return `${toX(i).toFixed(1)},${toY(avg).toFixed(1)}`;
        }).join(' ');
        return <polyline points={maLine} fill="none" stroke="#808000" strokeWidth="0.8" strokeDasharray="3 2" opacity="0.6" />;
      })()}
      {/* Confidence bands (1SD around forward test) */}
      {showConfBands && fwdPoints.length > 1 && (() => {
        const sdOffset = 2.5;
        const upperBand = fwdPoints.map((p, i) => `${toX(fwdIdx + i).toFixed(1)},${toY(p + sdOffset).toFixed(1)}`).join(' ');
        const lowerBand = fwdPoints.map((p, i) => `${toX(fwdIdx + i).toFixed(1)},${toY(p - sdOffset).toFixed(1)}`).join(' ');
        return (
          <>
            <polyline points={upperBand} fill="none" stroke={EQ_BT_COLOR} strokeWidth="0.5" strokeDasharray="2 2" opacity="0.3" />
            <polyline points={lowerBand} fill="none" stroke={EQ_BT_COLOR} strokeWidth="0.5" strokeDasharray="2 2" opacity="0.3" />
          </>
        );
      })()}
      {/* BT gradient fill */}
      <polygon points={buildFill(btPoints.slice(0, fwdIdx + 1), 0)} fill={`url(#btG${seed})`} />
      {/* FWD gradient fill */}
      <polygon points={buildFill(fwdPoints, fwdIdx)} fill={`url(#fwG${seed})`} />
      {/* FWD boundary */}
      <line x1={bndX} y1="0" x2={bndX} y2={h} stroke={EQ_FWD_COLOR} strokeWidth="0.5" strokeDasharray="3 2" opacity="0.5" />
      {/* BT line */}
      <polyline points={buildLine(btPoints.slice(0, fwdIdx + 1), 0)} fill="none" stroke={EQ_BT_COLOR} strokeWidth="1.5" />
      {/* FWD line */}
      <polyline points={buildLine(fwdPoints, fwdIdx)} fill="none" stroke={EQ_FWD_COLOR} strokeWidth="1.5" />
      {/* Live alerts — overlaid on FWD x-range, shows slippage */}
      {livePoints.length > 0 && <polyline points={buildLine(livePoints, fwdIdx)} fill="none" stroke={EQ_LIVE_COLOR} strokeWidth="1.5" />}
    </svg>
  );
}

// Mock SD values per strategy (would be computed from forward test vs backtest distribution)
function getStrategySD(stratId: string): { fwd: number; alert: number } {
  const fwdMap: Record<string, number> = { '1': 0.8, '2': -0.3, '3': 1.5, '4': 2.1, '5': 1.9 };
  const alertMap: Record<string, number> = { '1': 0.6, '2': -0.5, '3': 1.2, '4': 2.4, '5': 2.1 };
  return { fwd: fwdMap[stratId] ?? 0, alert: alertMap[stratId] ?? 0 };
}

/* ========================================================================= */
/* COMPONENT                                                                   */
/* ========================================================================= */

const PULSE_CSS = `@keyframes pulse { 0%, 100% { transform: scale(1); opacity: 0.5; } 50% { transform: scale(2.2); opacity: 0; } }`;

interface StrategiesV5Props {
  /** When provided, replaces mockStrategies with real API data */
  apiStrategies?: Strategy[];
  /** Called when a strategy is deleted */
  onDelete?: (id: number) => void;
  /** Called when a strategy is duplicated */
  onDuplicate?: (id: number) => void;
  /** Called on bulk delete */
  onBulkDelete?: (ids: number[]) => void;
}

export default function StrategiesV5({
  apiStrategies,
  onDelete,
  onDuplicate,
  onBulkDelete,
}: StrategiesV5Props = {}) {
  useEffect(() => {
    const id = 'strategies-pulse-css';
    if (!document.getElementById(id)) {
      const s = document.createElement('style'); s.id = id; s.textContent = PULSE_CSS; document.head.appendChild(s);
    }
  }, []);

  // Use API data when provided, fall back to mocks
  const strategies = apiStrategies ?? mockStrategies;

  // Filter state
  const [tickerFilter, setTickerFilter] = useState('All');
  const [directionFilter, setDirectionFilter] = useState('All');
  const [tagFilter, setTagFilter] = useState('All');
  const [statusFilter, setStatusFilter] = useState('All');
  const [tqFilter, setTqFilter] = useState('None');
  const [dataView, setDataView] = useState('Strategy Default');
  const [sortBy, setSortBy] = useState('Newest First');
  const [eqXAxis, setEqXAxis] = useState<'time' | 'trade'>('time');
  const [eqShowHWM, setEqShowHWM] = useState(true);
  const [eqShowEdgeMA, setEqShowEdgeMA] = useState(false);
  const [eqShowConfBands, setEqShowConfBands] = useState(false);
  const [kpiMode, setKpiMode] = useState('Overall');
  const [chartHeight, setChartHeight] = useState(64);

  // Bulk select (always available, no select mode toggle needed)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  // Delete confirmation
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [bulkDeleteConfirm, setBulkDeleteConfirm] = useState(false);

  // Derived filter options
  const allTickers = useMemo(() => Array.from(new Set(strategies.map((s) => s.symbol))).sort(), [strategies]);
  const allTags = useMemo(() => Array.from(new Set(strategies.flatMap((s) => s.tags))).sort(), [strategies]);
  const allStatuses = ['On Track', 'Outperforming', 'Underperforming', 'Insufficient Data'];

  // Filtered + sorted strategies
  const filteredStrategies = useMemo(() => {
    let result = [...strategies];
    if (tickerFilter !== 'All') result = result.filter((s) => s.symbol === tickerFilter);
    if (directionFilter !== 'All') result = result.filter((s) => s.direction === directionFilter);
    if (tagFilter !== 'All') result = result.filter((s) => s.tags.includes(tagFilter));
    if (statusFilter !== 'All') result = result.filter((s) => s.status === statusFilter);

    // Sort
    switch (sortBy) {
      case 'Newest First': result.sort((a, b) => parseInt(b.id) - parseInt(a.id)); break;
      case 'Oldest First': result.sort((a, b) => parseInt(a.id) - parseInt(b.id)); break;
      case 'Name A-Z': result.sort((a, b) => a.name.localeCompare(b.name)); break;
      case 'Win Rate (High)': result.sort((a, b) => b.winRate - a.winRate); break;
      case 'Profit Factor (High)': result.sort((a, b) => b.pf - a.pf); break;
      case 'Daily R (High)': result.sort((a, b) => b.dailyR - a.dailyR); break;
      case 'Max DD (Best)': result.sort((a, b) => b.maxDD - a.maxDD); break;
    }
    return result;
  }, [tickerFilter, directionFilter, tagFilter, statusFilter, sortBy]);

  const toggleSelect = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const selectAll = () => {
    setSelectedIds(new Set(filteredStrategies.map((s) => s.id)));
  };

  const clearSelection = () => {
    setSelectedIds(new Set());
  };

  const selectStyle = {
    background: 'var(--bg-input)',
    border: '1px solid var(--border)',
    color: 'var(--text-primary)',
    padding: '6px 12px',
    borderRadius: '8px',
    fontSize: '0.875rem',
  };

  const btnSecondary = {
    background: 'var(--bg-card)',
    border: '1px solid var(--border)',
    color: 'var(--text-secondary)',
    padding: '6px 14px',
    borderRadius: '8px',
    fontSize: '0.875rem',
    cursor: 'pointer' as const,
  };

  const btnPrimary = {
    background: 'var(--accent)',
    border: 'none',
    color: 'white',
    padding: '6px 14px',
    borderRadius: '8px',
    fontSize: '0.875rem',
    fontWeight: 500,
    cursor: 'pointer' as const,
  };

  return (
    <div>
      {/* ---- Header ---- */}
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">My Strategies</h1>
        <div className="flex gap-3">
          <button style={btnSecondary}>Reset All Alerts</button>
          <button style={btnSecondary}>Update Data</button>
          <button style={btnPrimary}>+ New Strategy</button>
        </div>
      </div>

      {/* ---- Bulk action bar (shows when any cards are checked) ---- */}
      {selectedIds.size > 0 && (
        <div
          className="flex items-center gap-4 mb-4 px-4 py-3 rounded-lg"
          style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}
        >
          <span className="text-sm" style={{ color: 'var(--text-muted)' }}>
            {selectedIds.size} selected
          </span>
          <button
            className="text-sm px-3 py-1 rounded"
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)', cursor: 'pointer' }}
            onClick={selectAll}
          >
            Select All
          </button>
          <button
            className="text-sm px-3 py-1 rounded"
            style={{ background: 'var(--red-muted)', color: 'var(--red)', border: 'none', cursor: 'pointer' }}
            onClick={() => setBulkDeleteConfirm(true)}
          >
            Delete Selected
          </button>
          <button style={btnPrimary} className="text-sm">
            Create Portfolio
          </button>
          <button style={btnSecondary} className="text-sm">
            Update Portfolio
          </button>
          <button style={btnSecondary} className="text-sm">
            Add Tag
          </button>
          <span className="flex-1" />
          <button
            className="text-sm px-3 py-1 rounded"
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-muted)', cursor: 'pointer' }}
            onClick={clearSelection}
          >
            Clear
          </button>
        </div>
      )}

      {/* ---- Bulk delete confirmation modal ---- */}
      <Modal
        title="Delete Selected Strategies"
        isOpen={bulkDeleteConfirm}
        onClose={() => setBulkDeleteConfirm(false)}
        width="480px"
      >
        <p className="text-sm mb-4" style={{ color: 'var(--text-secondary)' }}>
          Delete {selectedIds.size} strategies? This cannot be undone.
        </p>
        <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>
          {strategies
            .filter((s) => selectedIds.has(s.id))
            .map((s) => s.name)
            .join(', ')}
        </p>
        <div className="flex gap-3 justify-end">
          <button style={btnSecondary} onClick={() => setBulkDeleteConfirm(false)}>Cancel</button>
          <button
            style={{ ...btnPrimary, background: 'var(--red)' }}
            onClick={() => { setBulkDeleteConfirm(false); clearSelection(); }}
          >
            Yes, Delete All
          </button>
        </div>
      </Modal>

      {/* ---- Filters ---- */}
      <div className="grid grid-cols-3 lg:grid-cols-7 gap-3 mb-5">
        <select style={selectStyle} value={tickerFilter} onChange={(e) => setTickerFilter(e.target.value)}>
          <option value="All">Ticker: All</option>
          {allTickers.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>
        <select style={selectStyle} value={directionFilter} onChange={(e) => setDirectionFilter(e.target.value)}>
          <option value="All">Direction: All</option>
          <option value="LONG">LONG</option>
          <option value="SHORT">SHORT</option>
        </select>
        <select style={selectStyle} value={tagFilter} onChange={(e) => setTagFilter(e.target.value)}>
          <option value="All">Tag: All</option>
          {allTags.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>
        <select style={selectStyle} value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
          <option value="All">Status: All</option>
          {allStatuses.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
        <select style={selectStyle} value={tqFilter} onChange={(e) => setTqFilter(e.target.value)}>
          <option value="None">TQ: None</option>
          <option value="ttp">TQ: Trade The Pool</option>
          <option value="ftmo">TQ: FTMO</option>
          <option value="topstep">TQ: Topstep</option>
          <option value="custom">TQ: My Custom Rules</option>
        </select>
        <select style={selectStyle} value={dataView} onChange={(e) => setDataView(e.target.value)}>
          {['Strategy Default', 'All Data', 'Last 7 Days', 'Last 30 Days', 'Last 90 Days', 'Backtest Only', 'Forward Test Only'].map((v) => (
            <option key={v} value={v}>{v}</option>
          ))}
        </select>
        <select style={selectStyle} value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
          {['Newest First', 'Oldest First', 'Name A-Z', 'Win Rate (High)', 'Profit Factor (High)', 'Daily R (High)', 'Max DD (Best)'].map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </div>

      {/* Viewing preferences row */}
      <div className="flex items-center gap-4 mb-4 text-[10px]" style={{ color: 'var(--text-muted)' }}>
        <div className="flex items-center gap-1.5">
          <span>KPIs:</span>
          <select
            className="px-2 py-1 rounded text-[10px] font-medium"
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
            value={kpiMode}
            onChange={(e) => setKpiMode(e.target.value)}
          >
            <option value="Overall">Overall</option>
            <option value="BT vs FWD">Backtest vs Forward</option>
            <option value="FWD vs Alerts">Forward vs Alerts</option>
            <option value="BT vs Alerts">Backtest vs Alerts</option>
          </select>
        </div>
        <div className="w-px h-4" style={{ background: 'var(--border)' }} />
        <div className="flex items-center gap-1.5">
          <span>Chart Height:</span>
          {([
            { value: 48, label: 'S' },
            { value: 64, label: 'M' },
            { value: 96, label: 'L' },
            { value: 140, label: 'XL' },
          ]).map((opt) => (
            <button
              key={opt.value}
              onClick={() => setChartHeight(opt.value)}
              className="px-2 py-1 rounded font-medium"
              style={{
                background: chartHeight === opt.value ? 'var(--accent-muted)' : 'var(--bg-input)',
                color: chartHeight === opt.value ? 'var(--accent)' : 'var(--text-muted)',
                border: chartHeight === opt.value ? '1px solid var(--accent)' : '1px solid var(--border)',
              }}
            >
              {opt.label}
            </button>
          ))}
        </div>
        <div className="w-px h-4" style={{ background: 'var(--border)' }} />
        <div className="flex items-center gap-1.5">
          <span>X-axis:</span>
          {(['time', 'trade'] as const).map((mode) => (
            <button key={mode} onClick={() => setEqXAxis(mode)} className="px-2 py-1 rounded font-medium" style={{ background: eqXAxis === mode ? 'var(--accent-muted)' : 'var(--bg-input)', color: eqXAxis === mode ? 'var(--accent)' : 'var(--text-muted)', border: eqXAxis === mode ? '1px solid var(--accent)' : '1px solid var(--border)' }}>
              {mode === 'time' ? 'Time' : 'Trade #'}
            </button>
          ))}
        </div>
        <div className="w-px h-4" style={{ background: 'var(--border)' }} />
        <div className="flex items-center gap-1.5">
          <span>High Water Mark:</span>
          {(['On', 'Off'] as const).map((v) => (
            <button key={v} onClick={() => setEqShowHWM(v === 'On')} className="px-2 py-1 rounded font-medium" style={{ background: (eqShowHWM ? 'On' : 'Off') === v ? 'var(--accent-muted)' : 'var(--bg-input)', color: (eqShowHWM ? 'On' : 'Off') === v ? 'var(--accent)' : 'var(--text-muted)', border: (eqShowHWM ? 'On' : 'Off') === v ? '1px solid var(--accent)' : '1px solid var(--border)' }}>
              {v}
            </button>
          ))}
        </div>
        <div className="w-px h-4" style={{ background: 'var(--border)' }} />
        <div className="flex items-center gap-1.5">
          <span>Edge Check:</span>
          {(['On', 'Off'] as const).map((v) => (
            <button key={v} onClick={() => setEqShowEdgeMA(v === 'On')} className="px-2 py-1 rounded font-medium" style={{ background: (eqShowEdgeMA ? 'On' : 'Off') === v ? 'var(--accent-muted)' : 'var(--bg-input)', color: (eqShowEdgeMA ? 'On' : 'Off') === v ? 'var(--accent)' : 'var(--text-muted)', border: (eqShowEdgeMA ? 'On' : 'Off') === v ? '1px solid var(--accent)' : '1px solid var(--border)' }}>
              {v}
            </button>
          ))}
        </div>
        <div className="w-px h-4" style={{ background: 'var(--border)' }} />
        <div className="flex items-center gap-1.5">
          <span>Confidence Bands:</span>
          {(['On', 'Off'] as const).map((v) => (
            <button key={v} onClick={() => setEqShowConfBands(v === 'On')} className="px-2 py-1 rounded font-medium" style={{ background: (eqShowConfBands ? 'On' : 'Off') === v ? 'var(--accent-muted)' : 'var(--bg-input)', color: (eqShowConfBands ? 'On' : 'Off') === v ? 'var(--accent)' : 'var(--text-muted)', border: (eqShowConfBands ? 'On' : 'Off') === v ? '1px solid var(--accent)' : '1px solid var(--border)' }}>
              {v}
            </button>
          ))}
        </div>
      </div>

      {/* ---- Strategy count ---- */}
      <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
        {filteredStrategies.length} strateg{filteredStrategies.length === 1 ? 'y' : 'ies'}
        {(tickerFilter !== 'All' || directionFilter !== 'All' || tagFilter !== 'All' || statusFilter !== 'All')
          ? ` (filtered from ${strategies.length})`
          : ''}
      </p>

      {/* ---- Empty state ---- */}
      {filteredStrategies.length === 0 && (
        <Card>
          <div className="text-center py-12">
            <p className="text-lg mb-2" style={{ color: 'var(--text-secondary)' }}>
              {strategies.length === 0
                ? 'No strategies yet. Create your first strategy!'
                : 'No strategies match the current filters.'}
            </p>
            {strategies.length === 0 && (
              <button style={btnPrimary}>Go to Strategy Builder</button>
            )}
          </div>
        </Card>
      )}

      {/* ---- Strategy cards (2-column grid) ---- */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {filteredStrategies.map((strat) => {
          const fwdDays = daysSince(strat.fwdSince);

          return (
            <Card key={strat.id}>
              {/* Name + status badge + monitored dot | SD top-right */}
              <div className="flex items-center gap-2 mb-1">
                {/* Pulsing dot if monitored */}
                {strat.monitored && (
                  <div className="relative flex-shrink-0 w-2.5 h-2.5">
                    <div className="w-2.5 h-2.5 rounded-full" style={{ background: 'var(--green)' }} />
                    <div className="w-2.5 h-2.5 rounded-full absolute top-0 left-0" style={{ background: 'var(--green)', animation: 'pulse 2s ease-out infinite', opacity: 0.5 }} />
                  </div>
                )}
                <h3 className="font-semibold text-base">{strat.name}</h3>
                <span
                  className="text-xs px-2 py-0.5 rounded-full font-medium"
                  style={{
                    color: statusColors[strat.status],
                    background: statusColors[strat.status] + '20',
                  }}
                >
                  {strat.status}
                </span>
                <span className="flex-1" />
                {strat.fwdTrades > 0 && (() => {
                  const { fwd, alert } = getStrategySD(strat.id);
                  const fmtSD = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}\u03c3`;
                  return (
                    <div className="flex items-center gap-1 flex-shrink-0">
                      <span className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded" style={{ color: EQ_FWD_COLOR, background: EQ_FWD_COLOR + '18' }}>
                        {fmtSD(fwd)}
                      </span>
                      {strat.alertTracking && (
                        <>
                          <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>|</span>
                          <span className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded" style={{ color: EQ_LIVE_COLOR, background: EQ_LIVE_COLOR + '18' }}>
                            {fmtSD(alert)}
                          </span>
                        </>
                      )}
                    </div>
                  );
                })()}
              </div>

              {/* Tags */}
              {strat.tags.length > 0 && (
                <div className="flex flex-wrap gap-1 mb-1" style={{ marginTop: '-2px' }}>
                  {strat.tags.map((tag) => (
                    <span
                      key={tag}
                      className="text-xs px-2 py-0.5 rounded-full"
                      style={{ background: 'var(--bg-input)', color: 'var(--text-muted)', fontSize: '0.72rem' }}
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              )}

              {/* Meta line: symbol | direction | session | BT days | Fwd days | alert accuracy */}
              <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
                {strat.symbol} {strat.direction}
                {strat.session !== 'RTH' && (
                  <span style={{ color: '#9C27B0' }}> | {strat.session}</span>
                )}
                <span style={{ color: EQ_BT_COLOR }}> | BT {strat.btDays}d</span>
                <span style={{ color: EQ_FWD_COLOR }}> | Fwd {fwdDays}d</span>
                {strat.alertTracking && (
                  <span style={{ color: EQ_LIVE_COLOR }}> | Alert Acc {(94 + parseInt(strat.id) * 0.7).toFixed(1)}%</span>
                )}
              </p>

              {/* Mini equity curve (3-segment per display settings V5) */}
              <div className="rounded-lg mb-2 overflow-hidden" style={{ background: 'var(--bg-input)' }}>
                <MiniEquityCurve
                  strategyId={strat.id}
                  fwdStartPct={strat.btDays / (strat.btDays + fwdDays)}
                  hasAlerts={strat.alertTracking}
                  showHWM={eqShowHWM}
                  showEdgeMA={eqShowEdgeMA}
                  showConfBands={eqShowConfBands}
                  height={chartHeight}
                />
              </div>
              {/* Legend */}
              <div className="flex gap-3 mb-3 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                <span><span style={{ color: EQ_BT_COLOR }}>{'\u2014'}</span> Backtest</span>
                <span><span style={{ color: EQ_FWD_COLOR }}>{'\u2014'}</span> Forward</span>
                {strat.alertTracking && <span><span style={{ color: EQ_LIVE_COLOR }}>{'\u2014'}</span> Alerts</span>}
              </div>

              {/* KPIs — mode-dependent */}
              {(() => {
                const fmt = (v: number | null, suffix = '') => v !== null ? `${v >= 0 && suffix !== '%' && suffix !== 'R' ? '+' : ''}${v.toFixed(suffix === '%' ? 1 : 2)}${suffix}` : '--';
                const fmtD = (a: number | null, b: number | null) => {
                  if (a === null || b === null) return '--';
                  const d = a - b;
                  const color = d > 0 ? 'var(--green)' : d < 0 ? 'var(--red)' : 'var(--text-muted)';
                  return <span style={{ color }}>{d >= 0 ? '+' : ''}{d.toFixed(1)}</span>;
                };

                const btTPD = strat.btDays > 0 ? strat.trades / strat.btDays : 0;
                const fwdTPD = fwdDays > 0 ? strat.fwdTrades / fwdDays : null;
                const alertTPD = strat.alertTrades > 0 && fwdDays > 0 ? strat.alertTrades / fwdDays : null;

                const btRow = { label: 'Backtest', color: EQ_BT_COLOR, wr: strat.winRate, pf: strat.pf, dr: strat.dailyR, roi: strat.dailyROI as number | null, tpd: btTPD as number | null, dd: strat.maxDD };
                const fwdRow = { label: 'Forward', color: EQ_FWD_COLOR, wr: strat.fwdWinRate, pf: strat.fwdPF, dr: null as number | null, roi: strat.fwdDailyROI, tpd: fwdTPD, dd: null as number | null };
                const alertRow = { label: 'Alerts', color: EQ_LIVE_COLOR, wr: strat.alertWinRate, pf: strat.alertPF, dr: strat.alertDailyR, roi: strat.alertDailyROI, tpd: alertTPD, dd: strat.alertMaxDD };

                if (kpiMode === 'Overall') {
                  return (
                    <div className="grid grid-cols-6 gap-2 mb-2">
                      {[
                        { label: 'WR', value: `${strat.winRate.toFixed(1)}%` },
                        { label: 'PF', value: strat.pf.toFixed(2) },
                        { label: 'Daily R', value: `${strat.dailyR >= 0 ? '+' : ''}${strat.dailyR.toFixed(2)}` },
                        { label: 'Daily ROI', value: `${strat.dailyROI >= 0 ? '+' : ''}${strat.dailyROI.toFixed(2)}%` },
                        { label: 'TPD', value: btTPD.toFixed(1) },
                        { label: 'Max DD', value: `${strat.maxDD.toFixed(1)}R` },
                      ].map((kpi, j) => (
                        <div key={j}>
                          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{kpi.label}</p>
                          <p className="text-sm font-medium">{kpi.value}</p>
                        </div>
                      ))}
                    </div>
                  );
                }

                // Comparison table mode
                const rowA = kpiMode === 'FWD vs Alerts' ? fwdRow : btRow;
                const rowB = kpiMode === 'BT vs FWD' ? fwdRow : alertRow;

                return (
                  <div className="mb-2 rounded-lg overflow-hidden border" style={{ borderColor: 'var(--border)' }}>
                    {/* Header */}
                    <div className="grid grid-cols-7 text-[10px] font-medium px-2 py-1.5" style={{ background: 'var(--bg-secondary)', color: 'var(--text-muted)' }}>
                      <span></span><span>WR</span><span>PF</span><span>Daily R</span><span>Daily ROI</span><span>TPD</span><span>Max DD</span>
                    </div>
                    {/* Row A */}
                    <div className="grid grid-cols-7 text-xs px-2 py-1.5 border-t" style={{ borderColor: 'var(--border)' }}>
                      <span className="text-[10px] font-medium" style={{ color: rowA.color }}>{rowA.label}</span>
                      <span>{rowA.wr !== null ? `${rowA.wr.toFixed(1)}%` : '--'}</span>
                      <span>{rowA.pf !== null ? rowA.pf.toFixed(2) : '--'}</span>
                      <span>{rowA.dr !== null ? fmt(rowA.dr) : '--'}</span>
                      <span>{rowA.roi !== null ? `${rowA.roi.toFixed(2)}%` : '--'}</span>
                      <span>{rowA.tpd !== null ? rowA.tpd.toFixed(1) : '--'}</span>
                      <span>{rowA.dd !== null ? `${rowA.dd.toFixed(1)}R` : '--'}</span>
                    </div>
                    {/* Row B */}
                    <div className="grid grid-cols-7 text-xs px-2 py-1.5 border-t" style={{ borderColor: 'var(--border)' }}>
                      <span className="text-[10px] font-medium" style={{ color: rowB.color }}>{rowB.label}</span>
                      <span>{rowB.wr !== null ? `${rowB.wr.toFixed(1)}%` : '--'}</span>
                      <span>{rowB.pf !== null ? rowB.pf.toFixed(2) : '--'}</span>
                      <span>{rowB.dr !== null ? fmt(rowB.dr) : '--'}</span>
                      <span>{rowB.roi !== null ? `${rowB.roi.toFixed(2)}%` : '--'}</span>
                      <span>{rowB.tpd !== null ? rowB.tpd.toFixed(1) : '--'}</span>
                      <span>{rowB.dd !== null ? `${rowB.dd.toFixed(1)}R` : '--'}</span>
                    </div>
                    {/* Delta row */}
                    <div className="grid grid-cols-7 text-xs px-2 py-1.5 border-t" style={{ borderColor: 'var(--border)', background: 'var(--bg-input)' }}>
                      <span className="text-[10px] font-medium" style={{ color: 'var(--text-muted)' }}>{'\u0394'}</span>
                      <span>{fmtD(rowB.wr, rowA.wr)}</span>
                      <span>{fmtD(rowB.pf, rowA.pf)}</span>
                      <span>{fmtD(rowB.dr, rowA.dr)}</span>
                      <span>{fmtD(rowB.roi, rowA.roi)}</span>
                      <span>{fmtD(rowB.tpd, rowA.tpd)}</span>
                      <span>{fmtD(rowB.dd, rowA.dd)}</span>
                    </div>
                  </div>
                );
              })()}

              {/* Strategy variables — pack-aware display */}
              <div className="space-y-1.5 mt-2">
                {/* Row 1: Entry */}
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>entry:</span>
                  {(() => {
                    const match = strat.entry.match(/^(\[[A-Z]+\])\s*(.+)$/);
                    const execTag = match?.[1] || '';
                    const rest = match?.[2] || strat.entry;
                    return (
                      <span className="text-[10px] font-mono" style={{ color: 'var(--text-secondary)' }}>
                        {execTag && <span className="font-medium px-1 py-0.5 rounded-full mr-1" style={{ color: EXEC_BADGE_COLOR, background: EXEC_BADGE_COLOR + '20', fontSize: '10px' }}>{execTag}</span>}
                        {rest}
                      </span>
                    );
                  })()}
                </div>
                {/* Row 2: Exit(s) */}
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>exit:</span>
                  {strat.exit.map((ex, i) => {
                    const match = ex.match(/^(\[[A-Z]+\])\s*(.+)$/);
                    const execTag = match?.[1] || '';
                    const rest = match?.[2] || ex;
                    return (
                      <span key={i} className="text-[10px] font-mono" style={{ color: 'var(--text-secondary)' }}>
                        {execTag && <span className="font-medium px-1 py-0.5 rounded-full mr-1" style={{ color: EXEC_BADGE_COLOR, background: EXEC_BADGE_COLOR + '20', fontSize: '10px' }}>{execTag}</span>}
                        {rest}
                        {i < strat.exit.length - 1 && <span style={{ color: 'var(--text-muted)' }}>{' , '}</span>}
                      </span>
                    );
                  })}
                </div>
                {/* Row 3: Stop + Target */}
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>stop:</span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded" style={{ color: 'var(--red)', background: 'var(--red)' + '18' }}>
                    {strat.stop}
                  </span>
                  <span className="text-[10px] ml-1" style={{ color: 'var(--text-muted)' }}>target:</span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded" style={{ color: 'var(--green)', background: 'var(--green)' + '18' }}>
                    {strat.target}
                  </span>
                </div>
                {/* Row 4: Confluence conditions with fidelity badges */}
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>confluence:</span>
                  {strat.confluence.length > 0 ? strat.confluence.map((c) => (
                    <span key={c} className="text-[10px] font-mono flex items-center gap-1">
                      <span className="px-1 py-0.5 rounded-full font-medium" style={{ color: '#26C6DA', background: '#26C6DA20', fontSize: '9px' }}>[PB]</span>
                      <span className="px-1.5 py-0.5 rounded" style={{ color: 'var(--accent)', background: 'var(--accent)' + '15' }}>{c}</span>
                    </span>
                  )) : (
                    <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>none</span>
                  )}
                </div>
              </div>

              {/* Action buttons + alert toggle + checkbox */}
              <div className="flex items-center gap-2 mt-2 pt-2" style={{ borderTop: '1px solid var(--border)' }}>
                <Link
                  href={`/strategies/${strat.id}`}
                  className="px-3 py-1.5 rounded text-xs"
                  style={{ ...btnSecondary, fontSize: '0.75rem', padding: '4px 12px' }}
                >
                  View
                </Link>
                <button className="px-3 py-1.5 rounded text-xs" style={{ ...btnSecondary, fontSize: '0.75rem', padding: '4px 12px' }}>
                  Edit
                </button>
                <button className="px-3 py-1.5 rounded text-xs" style={{ ...btnSecondary, fontSize: '0.75rem', padding: '4px 12px' }}>
                  Clone
                </button>
                <button
                  className="px-3 py-1.5 rounded text-xs"
                  style={{ background: 'var(--red-muted)', color: 'var(--red)', border: 'none', fontSize: '0.75rem', padding: '4px 12px', cursor: 'pointer' }}
                  onClick={() => setDeleteId(strat.id)}
                >
                  Delete
                </button>
                <button className="px-3 py-1.5 rounded text-xs" style={{ ...btnSecondary, fontSize: '0.75rem', padding: '4px 12px' }}>
                  Tags
                </button>
                <span className="flex-1" />
                {/* Alert tracking toggle */}
                <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Alerts</span>
                <div
                  className="relative w-7 h-4 rounded-full cursor-pointer flex-shrink-0"
                  style={{ background: strat.alertTracking ? 'var(--accent)' : 'var(--bg-input)', border: strat.alertTracking ? 'none' : '1px solid var(--border)' }}
                  title={strat.alertTracking ? 'Alerts on' : 'Alerts off'}
                >
                  <div className="absolute top-0.5 w-3 h-3 rounded-full transition-all" style={{ background: 'white', left: strat.alertTracking ? '12px' : '2px' }} />
                </div>
                <input
                  type="checkbox"
                  checked={selectedIds.has(strat.id)}
                  onChange={() => toggleSelect(strat.id)}
                  className="w-4 h-4 rounded cursor-pointer flex-shrink-0"
                  style={{ accentColor: 'var(--accent)' }}
                  title="Select for bulk actions"
                />
              </div>

              {/* Inline delete confirmation */}
              {deleteId === strat.id && (
                <div className="mt-3 p-3 rounded-lg" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                  <p className="text-sm mb-3" style={{ color: 'var(--text-secondary)' }}>
                    Delete &apos;{strat.name}&apos;? This cannot be undone.
                  </p>
                  <div className="flex gap-2">
                    <button
                      style={{ ...btnPrimary, background: 'var(--red)', fontSize: '0.75rem', padding: '4px 12px' }}
                      onClick={() => setDeleteId(null)}
                    >
                      Yes, Delete
                    </button>
                    <button
                      style={{ ...btnSecondary, fontSize: '0.75rem', padding: '4px 12px' }}
                      onClick={() => setDeleteId(null)}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}
            </Card>
          );
        })}
      </div>
    </div>
  );
}
