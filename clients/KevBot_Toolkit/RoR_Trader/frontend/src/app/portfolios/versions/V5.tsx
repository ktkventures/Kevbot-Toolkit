'use client';

import { useState, useMemo, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import Card from '@/components/Card';
import Modal from '@/components/Modal';
import ChartPlaceholder from '@/components/ChartPlaceholder';

/* ========================================================================= */
/* MOCK DATA                                                                  */
/* ========================================================================= */

interface Portfolio {
  id: string;
  name: string;
  status: 'On Track' | 'Outperforming' | 'Underperforming' | 'Insufficient Data';
  enabled: boolean;
  strategies: { symbol: string; direction: string }[];
  startingBalance: number;
  compoundRate: number;
  avgRiskPerTrade: number;
  webhookTemplate: string | null;
  requirementSet: string | null;
  reqPassing: number | null;
  reqTotal: number | null;
  // KPIs
  totalPnl: number;
  finalBalance: number;
  winRate: number;
  pf: number;
  maxDDPct: number;
  avgDailyPnl: number;
  trades: number;
  tradesPerDay: number;
  // Forward test
  fwdDays: number;
  fwdTotalPnl: number | null;
  fwdWinRate: number | null;
  fwdPF: number | null;
  fwdMaxDDPct: number | null;
  fwdAvgDailyPnl: number | null;
  // Alert/Live
  alertTotalPnl: number | null;
  alertWinRate: number | null;
  alertPF: number | null;
  alertMaxDDPct: number | null;
  alertAvgDailyPnl: number | null;
  // Sigma
  fwdSD: number;
  alertSD: number;
  // Meta
  tags: string[];
  createdAt: string;
  btDays: number;
}

const mockPortfolios: Portfolio[] = [
  {
    id: '1', name: 'Main Portfolio', status: 'On Track', enabled: true,
    strategies: [{ symbol: 'NVDA', direction: 'LONG' }, { symbol: 'SPY', direction: 'LONG' }, { symbol: 'TSLA', direction: 'LONG' }, { symbol: 'AAPL', direction: 'LONG' }],
    startingBalance: 10000, compoundRate: 0.02, avgRiskPerTrade: 250, webhookTemplate: 'SignalStack — Main Account',
    requirementSet: 'SIM1', reqPassing: 4, reqTotal: 5,
    totalPnl: 4250, finalBalance: 14250, winRate: 54.2, pf: 2.08, maxDDPct: -3.8, avgDailyPnl: 142, trades: 312, tradesPerDay: 3.5,
    fwdDays: 17, fwdTotalPnl: 820, fwdWinRate: 52.1, fwdPF: 1.85, fwdMaxDDPct: -2.9, fwdAvgDailyPnl: 48,
    alertTotalPnl: 760, alertWinRate: 50.5, alertPF: 1.72, alertMaxDDPct: -3.1, alertAvgDailyPnl: 45,
    fwdSD: 0.8, alertSD: 0.6, tags: ['Live', 'Prop Firm'],
    createdAt: '2026-02-15', btDays: 90,
  },
  {
    id: '2', name: 'Scalping Portfolio', status: 'Outperforming', enabled: true,
    strategies: [{ symbol: 'NVDA', direction: 'LONG' }, { symbol: 'TSLA', direction: 'LONG' }],
    startingBalance: 5000, compoundRate: 0, avgRiskPerTrade: 100, webhookTemplate: 'SignalStack — Main Account',
    requirementSet: 'SIM1', reqPassing: 5, reqTotal: 5,
    totalPnl: 2100, finalBalance: 7100, winRate: 56.8, pf: 2.45, maxDDPct: -2.1, avgDailyPnl: 70, trades: 198, tradesPerDay: 2.2,
    fwdDays: 17, fwdTotalPnl: 480, fwdWinRate: 58.3, fwdPF: 2.60, fwdMaxDDPct: -1.8, fwdAvgDailyPnl: 28,
    alertTotalPnl: 450, alertWinRate: 57.1, alertPF: 2.50, alertMaxDDPct: -1.9, alertAvgDailyPnl: 26,
    fwdSD: -0.3, alertSD: -0.5, tags: ['Live', 'Scalping'],
    createdAt: '2026-03-01', btDays: 60,
  },
  {
    id: '3', name: 'Swing Portfolio', status: 'Insufficient Data', enabled: false,
    strategies: [{ symbol: 'SPY', direction: 'LONG' }, { symbol: 'META', direction: 'LONG' }],
    startingBalance: 25000, compoundRate: 0.01, avgRiskPerTrade: 500, webhookTemplate: null,
    requirementSet: null, reqPassing: null, reqTotal: null,
    totalPnl: 1850, finalBalance: 26850, winRate: 48.5, pf: 1.42, maxDDPct: -5.2, avgDailyPnl: 62, trades: 67, tradesPerDay: 0.4,
    fwdDays: 6, fwdTotalPnl: null, fwdWinRate: null, fwdPF: null, fwdMaxDDPct: null, fwdAvgDailyPnl: null,
    alertTotalPnl: null, alertWinRate: null, alertPF: null, alertMaxDDPct: null, alertAvgDailyPnl: null,
    fwdSD: 1.5, alertSD: 0, tags: ['Swing'],
    createdAt: '2026-03-18', btDays: 150,
  },
  {
    id: '4', name: 'Paper Testing', status: 'Underperforming', enabled: true,
    strategies: [{ symbol: 'AAPL', direction: 'LONG' }, { symbol: 'META', direction: 'LONG' }, { symbol: 'TSLA', direction: 'LONG' }],
    startingBalance: 50000, compoundRate: 0, avgRiskPerTrade: 1000, webhookTemplate: 'SignalStack — Paper Account',
    requirementSet: 'SIM2', reqPassing: 3, reqTotal: 6,
    totalPnl: -1200, finalBalance: 48800, winRate: 44.8, pf: 0.92, maxDDPct: -6.5, avgDailyPnl: -40, trades: 145, tradesPerDay: 1.6,
    fwdDays: 19, fwdTotalPnl: -380, fwdWinRate: 42.0, fwdPF: 0.85, fwdMaxDDPct: -4.8, fwdAvgDailyPnl: -20,
    alertTotalPnl: -420, alertWinRate: 41.5, alertPF: 0.82, alertMaxDDPct: -5.1, alertAvgDailyPnl: -22,
    fwdSD: 2.1, alertSD: 2.4, tags: ['Paper', 'Testing'],
    createdAt: '2026-03-05', btDays: 90,
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

const EQ_BT_COLOR = '#2196F3';
const EQ_FWD_COLOR = '#FF9800';
const EQ_LIVE_COLOR = '#4CAF50';

const PULSE_CSS = `@keyframes pulse { 0%, 100% { transform: scale(1); opacity: 0.5; } 50% { transform: scale(2.2); opacity: 0; } }`;

function MiniEquityCurve({ portfolioId, fwdStartPct, hasAlerts, showHWM, showEdgeMA, showConfBands, height = 64 }: {
  portfolioId: string; fwdStartPct: number; hasAlerts: boolean;
  showHWM: boolean; showEdgeMA: boolean; showConfBands: boolean; height?: number;
}) {
  const seed = parseInt(portfolioId, 10) || 1;
  const totalPoints = 50;
  const fwdIdx = Math.max(1, Math.floor(totalPoints * fwdStartPct));
  const w = 320;
  const h = height;
  const pad = 3;

  const btPoints: number[] = [0];
  let val = 0;
  for (let i = 1; i <= totalPoints; i++) {
    val += Math.sin(seed * 137.5 * i) * 0.5 + 0.2;
    btPoints.push(val);
  }

  const fwdPoints = btPoints.slice(fwdIdx).map((p, i) => p + Math.sin(seed * 42 * (fwdIdx + i)) * 0.3);
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

  let hwmPeak = -Infinity;
  const hwmLine = btPoints.map((p, i) => { hwmPeak = Math.max(hwmPeak, p); return `${toX(i).toFixed(1)},${toY(hwmPeak).toFixed(1)}`; }).join(' ');

  const zeroY = toY(0);
  const bndX = toX(fwdIdx);

  return (
    <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" style={{ display: 'block' }}>
      <defs>
        <linearGradient id={`pbtG${seed}`} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={EQ_BT_COLOR} stopOpacity="0.2" /><stop offset="100%" stopColor={EQ_BT_COLOR} stopOpacity="0" /></linearGradient>
        <linearGradient id={`pfwG${seed}`} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={EQ_FWD_COLOR} stopOpacity="0.15" /><stop offset="100%" stopColor={EQ_FWD_COLOR} stopOpacity="0" /></linearGradient>
      </defs>
      <line x1="0" y1={zeroY} x2={w} y2={zeroY} stroke="var(--text-muted)" strokeWidth="0.5" strokeDasharray="4 3" opacity="0.4" />
      {showHWM && <polyline points={hwmLine} fill="none" stroke="var(--green)" strokeWidth="0.5" strokeDasharray="2 2" opacity="0.4" />}
      {showEdgeMA && (() => {
        const maLine = btPoints.map((_, i) => {
          const window = btPoints.slice(Math.max(0, i - 7), i + 1);
          const avg = window.reduce((s, v) => s + v, 0) / window.length;
          return `${toX(i).toFixed(1)},${toY(avg).toFixed(1)}`;
        }).join(' ');
        return <polyline points={maLine} fill="none" stroke="#808000" strokeWidth="0.8" strokeDasharray="3 2" opacity="0.6" />;
      })()}
      {showConfBands && fwdPoints.length > 1 && (() => {
        const sdOffset = 2.5;
        const upper = fwdPoints.map((p, i) => `${toX(fwdIdx + i).toFixed(1)},${toY(p + sdOffset).toFixed(1)}`).join(' ');
        const lower = fwdPoints.map((p, i) => `${toX(fwdIdx + i).toFixed(1)},${toY(p - sdOffset).toFixed(1)}`).join(' ');
        return (<><polyline points={upper} fill="none" stroke={EQ_BT_COLOR} strokeWidth="0.5" strokeDasharray="2 2" opacity="0.3" /><polyline points={lower} fill="none" stroke={EQ_BT_COLOR} strokeWidth="0.5" strokeDasharray="2 2" opacity="0.3" /></>);
      })()}
      <polygon points={buildFill(btPoints.slice(0, fwdIdx + 1), 0)} fill={`url(#pbtG${seed})`} />
      <polygon points={buildFill(fwdPoints, fwdIdx)} fill={`url(#pfwG${seed})`} />
      <line x1={bndX} y1="0" x2={bndX} y2={h} stroke={EQ_FWD_COLOR} strokeWidth="0.5" strokeDasharray="3 2" opacity="0.5" />
      <polyline points={buildLine(btPoints.slice(0, fwdIdx + 1), 0)} fill="none" stroke={EQ_BT_COLOR} strokeWidth="1.5" />
      <polyline points={buildLine(fwdPoints, fwdIdx)} fill="none" stroke={EQ_FWD_COLOR} strokeWidth="1.5" />
      {livePoints.length > 0 && <polyline points={buildLine(livePoints, fwdIdx)} fill="none" stroke={EQ_LIVE_COLOR} strokeWidth="1.5" />}
    </svg>
  );
}

/* ========================================================================= */
/* COMPONENT                                                                   */
/* ========================================================================= */

export default function PortfoliosV5() {
  const router = useRouter();
  useEffect(() => {
    const id = 'portfolios-pulse-css';
    if (!document.getElementById(id)) {
      const s = document.createElement('style'); s.id = id; s.textContent = PULSE_CSS; document.head.appendChild(s);
    }
  }, []);

  // Filters
  const [statusFilter, setStatusFilter] = useState('All');
  const [tagFilter, setTagFilter] = useState('All');
  const [tqFilter, setTqFilter] = useState('None');
  const [sortBy, setSortBy] = useState('Newest First');
  const [dataView, setDataView] = useState('All Data');
  const [customStart, setCustomStart] = useState('');
  const [customEnd, setCustomEnd] = useState('');
  const [kpiMode, setKpiMode] = useState('Overall');
  const [chartHeight, setChartHeight] = useState(64);
  const [eqXAxis, setEqXAxis] = useState<'time' | 'trade'>('time');
  const [eqShowHWM, setEqShowHWM] = useState(true);
  const [eqShowEdgeMA, setEqShowEdgeMA] = useState(false);
  const [eqShowConfBands, setEqShowConfBands] = useState(false);

  // Bulk select
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [bulkDeleteConfirm, setBulkDeleteConfirm] = useState(false);

  const allStatuses = ['On Track', 'Outperforming', 'Underperforming', 'Insufficient Data'];
  const allTags = useMemo(() => Array.from(new Set(mockPortfolios.flatMap((p) => p.tags))).sort(), []);

  const filteredPortfolios = useMemo(() => {
    let result = [...mockPortfolios];
    if (statusFilter !== 'All') result = result.filter((p) => p.status === statusFilter);
    if (tagFilter !== 'All') result = result.filter((p) => p.tags.includes(tagFilter));
    switch (sortBy) {
      case 'Newest First': result.sort((a, b) => parseInt(b.id) - parseInt(a.id)); break;
      case 'Oldest First': result.sort((a, b) => parseInt(a.id) - parseInt(b.id)); break;
      case 'Name A-Z': result.sort((a, b) => a.name.localeCompare(b.name)); break;
      case 'P&L (High)': result.sort((a, b) => b.totalPnl - a.totalPnl); break;
      case 'Win Rate (High)': result.sort((a, b) => b.winRate - a.winRate); break;
      case 'Balance (High)': result.sort((a, b) => b.finalBalance - a.finalBalance); break;
    }
    return result;
  }, [statusFilter, sortBy]);

  const toggleSelect = (id: string) => {
    setSelectedIds((prev) => { const next = new Set(prev); if (next.has(id)) next.delete(id); else next.add(id); return next; });
  };
  const selectAll = () => setSelectedIds(new Set(filteredPortfolios.map((p) => p.id)));
  const clearSelection = () => setSelectedIds(new Set());

  const selectStyle: React.CSSProperties = {
    background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)',
    padding: '6px 12px', borderRadius: '8px', fontSize: '0.875rem',
  };
  const btnSecondary: React.CSSProperties = {
    background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)',
    padding: '6px 14px', borderRadius: '8px', fontSize: '0.875rem', cursor: 'pointer',
  };
  const btnPrimary: React.CSSProperties = {
    background: 'var(--accent)', border: 'none', color: 'white',
    padding: '6px 14px', borderRadius: '8px', fontSize: '0.875rem', fontWeight: 500, cursor: 'pointer',
  };

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">My Portfolios</h1>
        <div className="flex gap-3">
          <button style={btnSecondary}>Update Data</button>
          <button style={btnPrimary} onClick={() => router.push('/portfolios/new')}>+ New Portfolio</button>
        </div>
      </div>

      {/* Bulk action bar */}
      {selectedIds.size > 0 && (
        <div className="flex items-center gap-4 mb-4 px-4 py-3 rounded-lg" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
          <span className="text-sm" style={{ color: 'var(--text-muted)' }}>{selectedIds.size} selected</span>
          <button className="text-sm px-3 py-1 rounded" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)', cursor: 'pointer' }} onClick={selectAll}>Select All</button>
          <button className="text-sm px-3 py-1 rounded" style={{ background: 'var(--red-muted)', color: 'var(--red)', border: 'none', cursor: 'pointer' }} onClick={() => setBulkDeleteConfirm(true)}>Delete Selected</button>
          <span className="flex-1" />
          <button className="text-sm px-3 py-1 rounded" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-muted)', cursor: 'pointer' }} onClick={clearSelection}>Clear</button>
        </div>
      )}

      {/* Bulk delete modal */}
      <Modal title="Delete Selected Portfolios" isOpen={bulkDeleteConfirm} onClose={() => setBulkDeleteConfirm(false)} width="480px">
        <p className="text-sm mb-4" style={{ color: 'var(--text-secondary)' }}>Delete {selectedIds.size} portfolios? This cannot be undone.</p>
        <div className="flex gap-3 justify-end">
          <button style={btnSecondary} onClick={() => setBulkDeleteConfirm(false)}>Cancel</button>
          <button style={{ ...btnPrimary, background: 'var(--red)' }} onClick={() => { setBulkDeleteConfirm(false); clearSelection(); }}>Yes, Delete All</button>
        </div>
      </Modal>

      {/* Filters */}
      <div className="grid grid-cols-2 lg:grid-cols-6 gap-3 mb-5">
        <select style={selectStyle} value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
          <option value="All">Status: All</option>
          {allStatuses.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
        <select style={selectStyle} value={tagFilter} onChange={(e) => setTagFilter(e.target.value)}>
          <option value="All">Tag: All</option>
          {allTags.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>
        <select style={selectStyle} value={tqFilter} onChange={(e) => setTqFilter(e.target.value)}>
          <option value="None">TQ: None</option>
          <option value="ttp">TQ: Trade The Pool</option>
          <option value="ftmo">TQ: FTMO</option>
          <option value="topstep">TQ: Topstep</option>
          <option value="custom">TQ: My Custom Rules</option>
        </select>
        <select style={selectStyle} value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
          {['Newest First', 'Oldest First', 'Name A-Z', 'P&L (High)', 'Win Rate (High)', 'Balance (High)'].map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
        <select style={selectStyle} value={dataView} onChange={(e) => setDataView(e.target.value)}>
          {['All Data', 'Last 7 Days', 'Last 30 Days', 'Last 90 Days', 'Backtest Only', 'Forward Only', 'Custom'].map((v) => (
            <option key={v} value={v}>{v}</option>
          ))}
        </select>
        {dataView === 'Custom' && (
          <>
            <input type="date" value={customStart} onChange={(e) => setCustomStart(e.target.value)} style={selectStyle} />
            <input type="date" value={customEnd} onChange={(e) => setCustomEnd(e.target.value)} style={selectStyle} />
          </>
        )}
      </div>

      {/* Viewing preferences */}
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
          {[{ value: 48, label: 'S' }, { value: 64, label: 'M' }, { value: 96, label: 'L' }, { value: 140, label: 'XL' }].map((opt) => (
            <button key={opt.value} onClick={() => setChartHeight(opt.value)} className="px-2 py-1 rounded font-medium"
              style={{ background: chartHeight === opt.value ? 'var(--accent-muted)' : 'var(--bg-input)', color: chartHeight === opt.value ? 'var(--accent)' : 'var(--text-muted)', border: chartHeight === opt.value ? '1px solid var(--accent)' : '1px solid var(--border)' }}>
              {opt.label}
            </button>
          ))}
        </div>
        <div className="w-px h-4" style={{ background: 'var(--border)' }} />
        <div className="flex items-center gap-1.5">
          <span>X-axis:</span>
          {(['time', 'trade'] as const).map((mode) => (
            <button key={mode} onClick={() => setEqXAxis(mode)} className="px-2 py-1 rounded font-medium"
              style={{ background: eqXAxis === mode ? 'var(--accent-muted)' : 'var(--bg-input)', color: eqXAxis === mode ? 'var(--accent)' : 'var(--text-muted)', border: eqXAxis === mode ? '1px solid var(--accent)' : '1px solid var(--border)' }}>
              {mode === 'time' ? 'Time' : 'Trade #'}
            </button>
          ))}
        </div>
        <div className="w-px h-4" style={{ background: 'var(--border)' }} />
        <div className="flex items-center gap-1.5">
          <span>High Water Mark:</span>
          {(['On', 'Off'] as const).map((v) => (
            <button key={v} onClick={() => setEqShowHWM(v === 'On')} className="px-2 py-1 rounded font-medium"
              style={{ background: (eqShowHWM ? 'On' : 'Off') === v ? 'var(--accent-muted)' : 'var(--bg-input)', color: (eqShowHWM ? 'On' : 'Off') === v ? 'var(--accent)' : 'var(--text-muted)', border: (eqShowHWM ? 'On' : 'Off') === v ? '1px solid var(--accent)' : '1px solid var(--border)' }}>
              {v}
            </button>
          ))}
        </div>
        <div className="w-px h-4" style={{ background: 'var(--border)' }} />
        <div className="flex items-center gap-1.5">
          <span>Edge Check:</span>
          {(['On', 'Off'] as const).map((v) => (
            <button key={v} onClick={() => setEqShowEdgeMA(v === 'On')} className="px-2 py-1 rounded font-medium"
              style={{ background: (eqShowEdgeMA ? 'On' : 'Off') === v ? 'var(--accent-muted)' : 'var(--bg-input)', color: (eqShowEdgeMA ? 'On' : 'Off') === v ? 'var(--accent)' : 'var(--text-muted)', border: (eqShowEdgeMA ? 'On' : 'Off') === v ? '1px solid var(--accent)' : '1px solid var(--border)' }}>
              {v}
            </button>
          ))}
        </div>
        <div className="w-px h-4" style={{ background: 'var(--border)' }} />
        <div className="flex items-center gap-1.5">
          <span>Confidence Bands:</span>
          {(['On', 'Off'] as const).map((v) => (
            <button key={v} onClick={() => setEqShowConfBands(v === 'On')} className="px-2 py-1 rounded font-medium"
              style={{ background: (eqShowConfBands ? 'On' : 'Off') === v ? 'var(--accent-muted)' : 'var(--bg-input)', color: (eqShowConfBands ? 'On' : 'Off') === v ? 'var(--accent)' : 'var(--text-muted)', border: (eqShowConfBands ? 'On' : 'Off') === v ? '1px solid var(--accent)' : '1px solid var(--border)' }}>
              {v}
            </button>
          ))}
        </div>
      </div>

      {/* Combined Metrics */}
      {filteredPortfolios.length > 0 && (() => {
        const combined = {
          totalPnl: filteredPortfolios.reduce((s, p) => s + p.totalPnl, 0),
          totalBalance: filteredPortfolios.reduce((s, p) => s + p.finalBalance, 0),
          startingBalance: filteredPortfolios.reduce((s, p) => s + p.startingBalance, 0),
          avgWR: filteredPortfolios.reduce((s, p) => s + p.winRate, 0) / filteredPortfolios.length,
          avgPF: filteredPortfolios.reduce((s, p) => s + p.pf, 0) / filteredPortfolios.length,
          worstDD: Math.min(...filteredPortfolios.map((p) => p.maxDDPct)),
          avgDailyPnl: filteredPortfolios.reduce((s, p) => s + p.avgDailyPnl, 0),
          totalTrades: filteredPortfolios.reduce((s, p) => s + p.trades, 0),
          totalStrategies: filteredPortfolios.reduce((s, p) => s + p.strategies.length, 0),
        };
        const roi = combined.startingBalance > 0 ? ((combined.totalPnl / combined.startingBalance) * 100) : 0;

        return (
          <Card className="mb-4">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-medium">
                Combined Metrics
                <span className="text-xs font-normal ml-2" style={{ color: 'var(--text-muted)' }}>
                  {filteredPortfolios.length} portfolio{filteredPortfolios.length !== 1 ? 's' : ''} &middot; {combined.totalStrategies} strategies
                  {(statusFilter !== 'All' || tagFilter !== 'All') && ` (filtered from ${mockPortfolios.length})`}
                </span>
              </h4>
            </div>

            {/* KPI row */}
            <div className="grid grid-cols-3 sm:grid-cols-5 lg:grid-cols-9 gap-3 mb-3">
              {[
                { label: 'Total P&L', value: `${combined.totalPnl >= 0 ? '+' : ''}$${Math.abs(combined.totalPnl).toLocaleString()}`, color: combined.totalPnl >= 0 ? 'var(--green)' : 'var(--red)' },
                { label: 'Combined Balance', value: `$${combined.totalBalance.toLocaleString()}` },
                { label: 'ROI', value: `${roi >= 0 ? '+' : ''}${roi.toFixed(1)}%`, color: roi >= 0 ? 'var(--green)' : 'var(--red)' },
                { label: 'Avg Win Rate', value: `${combined.avgWR.toFixed(1)}%` },
                { label: 'Avg PF', value: combined.avgPF.toFixed(2) },
                { label: 'Worst Max DD', value: `${combined.worstDD.toFixed(1)}%`, color: 'var(--red)' },
                { label: 'Combined Daily', value: `${combined.avgDailyPnl >= 0 ? '+' : ''}$${Math.abs(combined.avgDailyPnl).toLocaleString()}`, color: combined.avgDailyPnl >= 0 ? 'var(--green)' : 'var(--red)' },
                { label: 'Total Trades', value: String(combined.totalTrades) },
                { label: 'Strategies', value: String(combined.totalStrategies) },
              ].map((kpi) => (
                <div key={kpi.label}>
                  <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{kpi.label}</p>
                  <p className="text-sm font-bold" style={kpi.color ? { color: kpi.color } : undefined}>{kpi.value}</p>
                </div>
              ))}
            </div>

            {/* Combined equity curve */}
            <ChartPlaceholder label="Combined equity curve: sum of all visible portfolio equity curves with DD shading" height={100} />
          </Card>
        );
      })()}

      {/* Portfolio cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {filteredPortfolios.map((port) => {
          const fmtDollar = (v: number) => `${v >= 0 ? '+' : ''}$${Math.abs(v).toLocaleString()}`;
          const fmtPct = (v: number | null) => v !== null ? `${v.toFixed(1)}%` : '--';
          const fmtD = (a: number | null, b: number | null) => {
            if (a === null || b === null) return <span style={{ color: 'var(--text-muted)' }}>--</span>;
            const d = a - b;
            return <span style={{ color: d > 0 ? 'var(--green)' : d < 0 ? 'var(--red)' : 'var(--text-muted)' }}>{d >= 0 ? '+' : ''}{d.toFixed(1)}</span>;
          };

          const hasAlerts = port.alertTotalPnl !== null;

          return (
            <Card key={port.id}>
              {/* Name + status + enabled dot */}
              <div className="flex items-center gap-2 mb-1">
                {port.enabled && (
                  <div className="relative flex-shrink-0 w-2.5 h-2.5">
                    <div className="w-2.5 h-2.5 rounded-full" style={{ background: 'var(--green)' }} />
                    <div className="w-2.5 h-2.5 rounded-full absolute top-0 left-0" style={{ background: 'var(--green)', animation: 'pulse 2s ease-out infinite', opacity: 0.5 }} />
                  </div>
                )}
                <h3 className="font-semibold text-base">{port.name}</h3>
                <span className="text-xs px-2 py-0.5 rounded-full font-medium" style={{ color: statusColors[port.status], background: statusColors[port.status] + '20' }}>
                  {port.status}
                </span>
                <span className="flex-1" />
                {/* Sigma badges */}
                {port.fwdDays > 0 && (() => {
                  const fmtSD = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}\u03c3`;
                  return (
                    <div className="flex items-center gap-1 flex-shrink-0">
                      <span className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded" style={{ color: EQ_FWD_COLOR, background: EQ_FWD_COLOR + '18' }}>
                        {fmtSD(port.fwdSD)}
                      </span>
                      {hasAlerts && (
                        <>
                          <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>|</span>
                          <span className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded" style={{ color: EQ_LIVE_COLOR, background: EQ_LIVE_COLOR + '18' }}>
                            {fmtSD(port.alertSD)}
                          </span>
                        </>
                      )}
                    </div>
                  );
                })()}
              </div>

              {/* Tags */}
              {port.tags.length > 0 && (
                <div className="flex flex-wrap gap-1 mb-1" style={{ marginTop: '-2px' }}>
                  {port.tags.map((tag) => (
                    <span key={tag} className="text-xs px-2 py-0.5 rounded-full" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)', fontSize: '0.72rem' }}>
                      {tag}
                    </span>
                  ))}
                </div>
              )}

              {/* Meta line */}
              <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
                {port.strategies.length} strategies
                <span> | ${port.startingBalance.toLocaleString()} balance</span>
                {port.compoundRate > 0 && <span> | {(port.compoundRate * 100).toFixed(0)}% scaling</span>}
                <span> | ${port.avgRiskPerTrade} avg risk</span>
                <span> | {port.tradesPerDay.toFixed(1)} trades/day</span>
                {port.webhookTemplate && <span style={{ color: 'var(--accent)' }}> | {port.webhookTemplate}</span>}
              </p>

              {/* Strategy names */}
              <div className="flex flex-wrap gap-1.5 mb-2">
                {port.strategies.slice(0, 5).map((s, i) => (
                  <span key={i} className="text-[10px] px-2 py-0.5 rounded-full" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>
                    {s.symbol} {s.direction}
                  </span>
                ))}
                {port.strategies.length > 5 && (
                  <span className="text-[10px] px-2 py-0.5 rounded-full" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>
                    +{port.strategies.length - 5} more
                  </span>
                )}
              </div>

              {/* Equity curve */}
              <div className="rounded-lg mb-2 overflow-hidden" style={{ background: 'var(--bg-input)' }}>
                <MiniEquityCurve
                  portfolioId={port.id}
                  fwdStartPct={port.btDays / (port.btDays + port.fwdDays)}
                  hasAlerts={hasAlerts}
                  showHWM={eqShowHWM}
                  showEdgeMA={eqShowEdgeMA}
                  showConfBands={eqShowConfBands}
                  height={chartHeight}
                />
              </div>
              <div className="flex gap-3 mb-3 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                <span><span style={{ color: EQ_BT_COLOR }}>{'\u2014'}</span> Backtest</span>
                <span><span style={{ color: EQ_FWD_COLOR }}>{'\u2014'}</span> Forward</span>
                {hasAlerts && <span><span style={{ color: EQ_LIVE_COLOR }}>{'\u2014'}</span> Alerts</span>}
              </div>

              {/* KPIs */}
              {(() => {
                if (kpiMode === 'Overall') {
                  return (
                    <div className="grid grid-cols-6 gap-2 mb-2">
                      {[
                        { label: 'P&L', value: fmtDollar(port.totalPnl), color: port.totalPnl >= 0 ? 'var(--green)' : 'var(--red)' },
                        { label: 'WR', value: `${port.winRate.toFixed(1)}%` },
                        { label: 'PF', value: port.pf.toFixed(2) },
                        { label: 'Max DD', value: `${port.maxDDPct.toFixed(1)}%` },
                        { label: 'Avg Daily', value: fmtDollar(port.avgDailyPnl), color: port.avgDailyPnl >= 0 ? 'var(--green)' : 'var(--red)' },
                        { label: 'Trades', value: String(port.trades) },
                      ].map((kpi, j) => (
                        <div key={j}>
                          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{kpi.label}</p>
                          <p className="text-sm font-medium" style={kpi.color ? { color: kpi.color } : undefined}>{kpi.value}</p>
                        </div>
                      ))}
                    </div>
                  );
                }

                // Comparison mode
                const btRow = { label: 'Backtest', color: EQ_BT_COLOR, pnl: port.totalPnl, wr: port.winRate, pf: port.pf, dd: port.maxDDPct, daily: port.avgDailyPnl };
                const fwdRow = { label: 'Forward', color: EQ_FWD_COLOR, pnl: port.fwdTotalPnl, wr: port.fwdWinRate, pf: port.fwdPF, dd: port.fwdMaxDDPct, daily: port.fwdAvgDailyPnl };
                const alertRow = { label: 'Alerts', color: EQ_LIVE_COLOR, pnl: port.alertTotalPnl, wr: port.alertWinRate, pf: port.alertPF, dd: port.alertMaxDDPct, daily: port.alertAvgDailyPnl };

                const rowA = kpiMode === 'FWD vs Alerts' ? fwdRow : btRow;
                const rowB = kpiMode === 'BT vs FWD' ? fwdRow : alertRow;

                return (
                  <div className="mb-2 rounded-lg overflow-hidden border" style={{ borderColor: 'var(--border)' }}>
                    <div className="grid grid-cols-6 text-[10px] font-medium px-2 py-1.5" style={{ background: 'var(--bg-secondary)', color: 'var(--text-muted)' }}>
                      <span></span><span>P&L</span><span>WR</span><span>PF</span><span>Max DD</span><span>Avg Daily</span>
                    </div>
                    <div className="grid grid-cols-6 text-xs px-2 py-1.5 border-t" style={{ borderColor: 'var(--border)' }}>
                      <span className="text-[10px] font-medium" style={{ color: rowA.color }}>{rowA.label}</span>
                      <span>{rowA.pnl !== null ? fmtDollar(rowA.pnl) : '--'}</span>
                      <span>{fmtPct(rowA.wr)}</span>
                      <span>{rowA.pf !== null ? rowA.pf.toFixed(2) : '--'}</span>
                      <span>{fmtPct(rowA.dd)}</span>
                      <span>{rowA.daily !== null ? fmtDollar(rowA.daily) : '--'}</span>
                    </div>
                    <div className="grid grid-cols-6 text-xs px-2 py-1.5 border-t" style={{ borderColor: 'var(--border)' }}>
                      <span className="text-[10px] font-medium" style={{ color: rowB.color }}>{rowB.label}</span>
                      <span>{rowB.pnl !== null ? fmtDollar(rowB.pnl) : '--'}</span>
                      <span>{fmtPct(rowB.wr)}</span>
                      <span>{rowB.pf !== null ? rowB.pf.toFixed(2) : '--'}</span>
                      <span>{fmtPct(rowB.dd)}</span>
                      <span>{rowB.daily !== null ? fmtDollar(rowB.daily) : '--'}</span>
                    </div>
                    <div className="grid grid-cols-6 text-xs px-2 py-1.5 border-t" style={{ borderColor: 'var(--border)', background: 'var(--bg-input)' }}>
                      <span className="text-[10px] font-medium" style={{ color: 'var(--text-muted)' }}>{'\u0394'}</span>
                      <span>{fmtD(rowB.pnl, rowA.pnl)}</span>
                      <span>{fmtD(rowB.wr, rowA.wr)}</span>
                      <span>{fmtD(rowB.pf, rowA.pf)}</span>
                      <span>{fmtD(rowB.dd, rowA.dd)}</span>
                      <span>{fmtD(rowB.daily, rowA.daily)}</span>
                    </div>
                  </div>
                );
              })()}

              {/* Requirement set badge */}
              {port.requirementSet && (
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Reqs:</span>
                  <span className="text-[10px] px-2 py-0.5 rounded-full font-medium" style={{
                    color: port.reqPassing === port.reqTotal ? 'var(--green)' : 'var(--orange)',
                    background: port.reqPassing === port.reqTotal ? 'var(--green-muted)' : 'rgba(255,152,0,0.12)',
                  }}>
                    {port.requirementSet} ({port.reqPassing}/{port.reqTotal} pass)
                  </span>
                </div>
              )}

              {/* Action row */}
              <div className="flex items-center gap-2 mt-2 pt-2" style={{ borderTop: '1px solid var(--border)' }}>
                <Link href={`/portfolios/${port.id}`} className="px-3 py-1.5 rounded text-xs" style={{ ...btnSecondary, fontSize: '0.75rem', padding: '4px 12px', textDecoration: 'none', color: 'var(--text-secondary)' }}>
                  View
                </Link>
                <button className="px-3 py-1.5 rounded text-xs" style={{ ...btnSecondary, fontSize: '0.75rem', padding: '4px 12px' }}>Edit</button>
                <button className="px-3 py-1.5 rounded text-xs" style={{ ...btnSecondary, fontSize: '0.75rem', padding: '4px 12px' }}>Clone</button>
                <button className="px-3 py-1.5 rounded text-xs" style={{ background: 'var(--red-muted)', color: 'var(--red)', border: 'none', fontSize: '0.75rem', padding: '4px 12px', cursor: 'pointer' }}
                  onClick={() => setDeleteId(port.id)}>
                  Delete
                </button>
                <button className="px-3 py-1.5 rounded text-xs" style={{ ...btnSecondary, fontSize: '0.75rem', padding: '4px 12px' }}>Tags</button>
                <span className="flex-1" />
                {/* Portfolio enabled toggle */}
                <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Enabled</span>
                <div
                  className="relative w-7 h-4 rounded-full cursor-pointer flex-shrink-0"
                  style={{ background: port.enabled ? 'var(--accent)' : 'var(--bg-input)', border: port.enabled ? 'none' : '1px solid var(--border)' }}
                  title={port.enabled ? 'Portfolio on' : 'Portfolio off'}
                >
                  <div className="absolute top-0.5 w-3 h-3 rounded-full transition-all" style={{ background: 'white', left: port.enabled ? '12px' : '2px' }} />
                </div>
                <input
                  type="checkbox"
                  checked={selectedIds.has(port.id)}
                  onChange={() => toggleSelect(port.id)}
                  className="w-4 h-4 rounded cursor-pointer flex-shrink-0"
                  style={{ accentColor: 'var(--accent)' }}
                  title="Select for bulk actions"
                />
              </div>

              {/* Inline delete confirmation */}
              {deleteId === port.id && (
                <div className="mt-3 p-3 rounded-lg" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                  <p className="text-sm mb-3" style={{ color: 'var(--text-secondary)' }}>
                    Delete &apos;{port.name}&apos;? This cannot be undone.
                  </p>
                  <div className="flex gap-2">
                    <button style={{ ...btnPrimary, background: 'var(--red)', fontSize: '0.75rem', padding: '4px 12px' }} onClick={() => setDeleteId(null)}>Yes, Delete</button>
                    <button style={{ ...btnSecondary, fontSize: '0.75rem', padding: '4px 12px' }} onClick={() => setDeleteId(null)}>Cancel</button>
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
