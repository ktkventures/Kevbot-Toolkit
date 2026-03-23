'use client';

import { useState, useMemo } from 'react';
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
  trades: number;
  maxDD: number;
  fwdWinRate: number | null;
  fwdPF: number | null;
  fwdTrades: number;
  fwdSince: string;
  trigger: string;
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
    winRate: 54.0, pf: 2.05, dailyR: 1.95, trades: 224, maxDD: -2.5,
    fwdWinRate: 52.1, fwdPF: 1.82, fwdTrades: 18, fwdSince: '2026-03-07',
    trigger: '[C] EMA Bull Cross', stop: 'Swing (5 bars, $0.03)', target: '2R',
    confluence: ['5M-RVOL-HIGH', '1D-MACD_LINE-BULL'], alertTracking: true, monitored: true, btDays: 90,
  },
  {
    id: '2', name: 'SPY LONG - Mass #1', symbol: 'SPY', direction: 'LONG', timeframe: '5Min',
    session: 'RTH', status: 'Outperforming', tags: ['Swing'],
    winRate: 62.5, pf: 3.12, dailyR: 2.41, trades: 89, maxDD: -1.8,
    fwdWinRate: 65.0, fwdPF: 3.45, fwdTrades: 12, fwdSince: '2026-03-01',
    trigger: '[L0] VWAP Cross Above', stop: 'Swing (8 bars, $0.05)', target: '3R',
    confluence: ['1H-EMA_STACK-SML'], alertTracking: true, monitored: true, btDays: 120,
  },
  {
    id: '3', name: 'AAPL LONG - Mass #5', symbol: 'AAPL', direction: 'LONG', timeframe: '1Min',
    session: 'RTH', status: 'Insufficient Data', tags: ['New'],
    winRate: 48.2, pf: 1.45, dailyR: 0.82, trades: 156, maxDD: -3.1,
    fwdWinRate: null, fwdPF: null, fwdTrades: 3, fwdSince: '2026-03-18',
    trigger: '[C] MACD Bull Cross', stop: 'ATR (14, 2x)', target: '1.5R',
    confluence: ['5M-RVOL-HIGH', '1D-EMA_STACK-SML'], alertTracking: false, monitored: false, btDays: 60,
  },
  {
    id: '4', name: 'TSLA LONG - Mass #5', symbol: 'TSLA', direction: 'LONG', timeframe: '1Min',
    session: 'RTH', status: 'On Track', tags: ['Scalping'],
    winRate: 51.3, pf: 1.78, dailyR: 1.23, trades: 201, maxDD: -2.9,
    fwdWinRate: 50.0, fwdPF: 1.65, fwdTrades: 8, fwdSince: '2026-03-10',
    trigger: '[HM] UT Bot Buy', stop: 'Swing (5 bars, $0.10)', target: '2R',
    confluence: [], alertTracking: true, monitored: false, btDays: 90,
  },
  {
    id: '5', name: 'META LONG - Mass #13', symbol: 'META', direction: 'LONG', timeframe: '5Min',
    session: 'RTH', status: 'Underperforming', tags: ['Swing', 'Low DD'],
    winRate: 45.8, pf: 1.12, dailyR: 0.34, trades: 67, maxDD: -4.2,
    fwdWinRate: 40.0, fwdPF: 0.95, fwdTrades: 15, fwdSince: '2026-03-05',
    trigger: '[C] EMA Bull Cross', stop: 'Swing (10 bars, $0.05)', target: '2.5R',
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
function MiniEquityCurve({ strategyId, color }: { strategyId: string; color: string }) {
  // Deterministic random walk based on id
  const seed = parseInt(strategyId, 10) || 1;
  const points: number[] = [0];
  let val = 0;
  for (let i = 1; i <= 40; i++) {
    const r = Math.sin(seed * 137.5 * i) * 0.5 + 0.2;
    val += r;
    points.push(val);
  }
  const min = Math.min(...points);
  const max = Math.max(...points);
  const range = max - min || 1;
  const w = 320;
  const h = 48;
  const pathData = points
    .map((p, i) => {
      const x = (i / (points.length - 1)) * w;
      const y = h - ((p - min) / range) * (h - 4) - 2;
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');

  return (
    <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" style={{ display: 'block' }}>
      <path d={pathData} fill="none" stroke={color} strokeWidth="2" />
    </svg>
  );
}

/* ========================================================================= */
/* COMPONENT                                                                   */
/* ========================================================================= */

export default function StrategiesV2() {
  // Filter state
  const [tickerFilter, setTickerFilter] = useState('All');
  const [directionFilter, setDirectionFilter] = useState('All');
  const [tagFilter, setTagFilter] = useState('All');
  const [statusFilter, setStatusFilter] = useState('All');
  const [sortBy, setSortBy] = useState('Newest First');

  // Bulk select
  const [selectMode, setSelectMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  // Delete confirmation
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [bulkDeleteConfirm, setBulkDeleteConfirm] = useState(false);

  // Derived filter options
  const allTickers = useMemo(() => Array.from(new Set(mockStrategies.map((s) => s.symbol))).sort(), []);
  const allTags = useMemo(() => Array.from(new Set(mockStrategies.flatMap((s) => s.tags))).sort(), []);
  const allStatuses = ['On Track', 'Outperforming', 'Underperforming', 'Insufficient Data'];

  // Filtered + sorted strategies
  const filteredStrategies = useMemo(() => {
    let result = [...mockStrategies];
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
    setSelectMode(false);
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

      {/* ---- Multi-select toggle ---- */}
      <div className="flex justify-end mb-3">
        <button
          style={selectMode ? btnPrimary : btnSecondary}
          onClick={() => { if (selectMode) clearSelection(); else setSelectMode(true); }}
        >
          {selectMode ? 'Cancel Selection' : 'Select'}
        </button>
      </div>

      {/* ---- Bulk action bar ---- */}
      {selectMode && selectedIds.size > 0 && (
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
          {mockStrategies
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
      <div className="grid grid-cols-5 gap-4 mb-5">
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
        <select style={selectStyle} value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
          {['Newest First', 'Oldest First', 'Name A-Z', 'Win Rate (High)', 'Profit Factor (High)', 'Daily R (High)', 'Max DD (Best)'].map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </div>

      {/* ---- Strategy count ---- */}
      <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
        {filteredStrategies.length} strateg{filteredStrategies.length === 1 ? 'y' : 'ies'}
        {(tickerFilter !== 'All' || directionFilter !== 'All' || tagFilter !== 'All' || statusFilter !== 'All')
          ? ` (filtered from ${mockStrategies.length})`
          : ''}
      </p>

      {/* ---- Empty state ---- */}
      {filteredStrategies.length === 0 && (
        <Card>
          <div className="text-center py-12">
            <p className="text-lg mb-2" style={{ color: 'var(--text-secondary)' }}>
              {mockStrategies.length === 0
                ? 'No strategies yet. Create your first strategy!'
                : 'No strategies match the current filters.'}
            </p>
            {mockStrategies.length === 0 && (
              <button style={btnPrimary}>Go to Strategy Builder</button>
            )}
          </div>
        </Card>
      )}

      {/* ---- Strategy cards (2-column grid) ---- */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {filteredStrategies.map((strat) => {
          const fwdDays = daysSince(strat.fwdSince);
          const equityCurveColor = strat.status === 'Outperforming' ? 'var(--blue)'
            : strat.status === 'Underperforming' ? 'var(--red)'
            : 'var(--green)';

          return (
            <Card key={strat.id}>
              {/* Selection checkbox */}
              {selectMode && (
                <label className="flex items-center gap-2 mb-2 cursor-pointer text-sm" style={{ color: 'var(--text-secondary)' }}>
                  <input
                    type="checkbox"
                    checked={selectedIds.has(strat.id)}
                    onChange={() => toggleSelect(strat.id)}
                    style={{ accentColor: 'var(--accent)' }}
                  />
                  Select
                </label>
              )}

              {/* Name + status badge */}
              <div className="flex items-center gap-3 mb-1">
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
                {strat.monitored && (
                  <span className="text-xs px-2 py-0.5 rounded-full" style={{ color: 'var(--orange)', background: 'var(--orange)' + '20' }}>
                    Monitored
                  </span>
                )}
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

              {/* Meta line: symbol | direction | session | BT days | Fwd days */}
              <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
                {strat.symbol} {strat.direction}
                {strat.session !== 'RTH' && (
                  <span style={{ color: '#9C27B0' }}> | {strat.session}</span>
                )}
                <span style={{ color: 'var(--blue)' }}> | BT {strat.btDays}d</span>
                <span style={{ color: 'var(--orange)' }}> | Fwd {fwdDays}d</span>
              </p>

              {/* Mini equity curve */}
              <div className="rounded-lg mb-3 overflow-hidden" style={{ background: 'var(--bg-input)' }}>
                <MiniEquityCurve strategyId={strat.id} color={equityCurveColor} />
              </div>

              {/* Primary KPIs */}
              <div className="grid grid-cols-5 gap-3 mb-2">
                {[
                  { label: 'WR', value: `${strat.winRate.toFixed(1)}%` },
                  { label: 'PF', value: strat.pf.toFixed(2) },
                  { label: 'Daily R', value: `${strat.dailyR >= 0 ? '+' : ''}${strat.dailyR.toFixed(2)}` },
                  { label: 'Trades', value: String(strat.trades) },
                  { label: 'Max DD', value: `${strat.maxDD.toFixed(1)}R` },
                ].map((kpi, j) => (
                  <div key={j}>
                    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{kpi.label}</p>
                    <p className="text-sm font-medium">{kpi.value}</p>
                  </div>
                ))}
              </div>

              {/* Forward test KPIs */}
              {(strat.fwdWinRate !== null || strat.fwdTrades > 0) && (
                <div className="grid grid-cols-5 gap-3 mb-2 pt-2" style={{ borderTop: '1px solid var(--border)' }}>
                  <div>
                    <p className="text-xs" style={{ color: 'var(--orange)' }}>Fwd WR</p>
                    <p className="text-sm font-medium">{strat.fwdWinRate !== null ? `${strat.fwdWinRate.toFixed(1)}%` : '--'}</p>
                  </div>
                  <div>
                    <p className="text-xs" style={{ color: 'var(--orange)' }}>Fwd PF</p>
                    <p className="text-sm font-medium">{strat.fwdPF !== null ? strat.fwdPF.toFixed(2) : '--'}</p>
                  </div>
                  <div>
                    <p className="text-xs" style={{ color: 'var(--orange)' }}>Fwd Trades</p>
                    <p className="text-sm font-medium">{strat.fwdTrades}</p>
                  </div>
                  <div className="col-span-2">
                    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Forward since</p>
                    <p className="text-sm">{strat.fwdSince}</p>
                  </div>
                </div>
              )}

              {/* Trigger / Stop / Confluence badges */}
              <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                Entry: {strat.trigger} | Stop: {strat.stop} | Target: {strat.target}
              </p>
              {strat.confluence.length > 0 ? (
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  Confluence: {strat.confluence.join(', ')}
                </p>
              ) : (
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Confluence: None</p>
              )}

              {/* Alert tracking toggle */}
              <div className="flex items-center gap-2 mt-2 mb-2">
                <label className="text-xs cursor-pointer flex items-center gap-2" style={{ color: 'var(--text-muted)' }}>
                  <div
                    className="relative inline-block w-8 h-4 rounded-full transition-colors"
                    style={{ background: strat.alertTracking ? 'var(--accent)' : 'var(--bg-input)' }}
                  >
                    <div
                      className="absolute top-0.5 w-3 h-3 rounded-full transition-transform"
                      style={{
                        background: 'white',
                        left: strat.alertTracking ? '16px' : '2px',
                      }}
                    />
                  </div>
                  Alert Tracking
                </label>
              </div>

              {/* Action buttons */}
              <div className="flex gap-2 mt-2 pt-2" style={{ borderTop: '1px solid var(--border)' }}>
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
