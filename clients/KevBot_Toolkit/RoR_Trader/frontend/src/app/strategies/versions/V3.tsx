'use client';

import { useState, useMemo } from 'react';
import Link from 'next/link';
import Card from '@/components/Card';
import Modal from '@/components/Modal';
import PageHeader from '@/components/PageHeader';

/* ========================================================================= */
/* MOCK DATA                                                                  */
/* ========================================================================= */

interface Strategy {
  id: string;
  name: string;
  symbol: string;
  timeframe: string;
  status: 'On Track' | 'Outperforming' | 'Underperforming' | 'Insufficient Data';
  winRate: number;
  pf: number;
  dailyR: number;
  trades: number;
  fwdStatus: string;
  monitored: boolean;
}

const mockStrategies: Strategy[] = [
  { id: '1', name: 'NVDA LONG - Mass #2', symbol: 'NVDA', timeframe: '1Min', status: 'On Track', winRate: 54.0, pf: 2.05, dailyR: 1.95, trades: 224, fwdStatus: '18 trades / 14d', monitored: true },
  { id: '2', name: 'SPY LONG - Mass #1', symbol: 'SPY', timeframe: '5Min', status: 'Outperforming', winRate: 62.5, pf: 3.12, dailyR: 2.41, trades: 89, fwdStatus: '12 trades / 20d', monitored: true },
  { id: '3', name: 'AAPL LONG - Mass #5', symbol: 'AAPL', timeframe: '1Min', status: 'Insufficient Data', winRate: 48.2, pf: 1.45, dailyR: 0.82, trades: 156, fwdStatus: '3 trades / 3d', monitored: false },
  { id: '4', name: 'TSLA LONG - Mass #5', symbol: 'TSLA', timeframe: '1Min', status: 'On Track', winRate: 51.3, pf: 1.78, dailyR: 1.23, trades: 201, fwdStatus: '8 trades / 11d', monitored: false },
  { id: '5', name: 'META LONG - Mass #13', symbol: 'META', timeframe: '5Min', status: 'Underperforming', winRate: 45.8, pf: 1.12, dailyR: 0.34, trades: 67, fwdStatus: '15 trades / 16d', monitored: false },
];

/* ========================================================================= */
/* HELPERS                                                                    */
/* ========================================================================= */

const statusDotColor: Record<string, string> = {
  'On Track': 'var(--green)',
  'Outperforming': 'var(--blue)',
  'Underperforming': 'var(--red)',
  'Insufficient Data': 'var(--text-muted)',
};

type SortKey = 'name' | 'symbol' | 'timeframe' | 'status' | 'winRate' | 'pf' | 'dailyR' | 'trades';
type SortDir = 'asc' | 'desc';

/* ========================================================================= */
/* COMPONENT                                                                  */
/* ========================================================================= */

export default function StrategiesV3() {
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState('All');
  const [sortKey, setSortKey] = useState<SortKey>('name');
  const [sortDir, setSortDir] = useState<SortDir>('asc');
  const [deleteId, setDeleteId] = useState<string | null>(null);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir('asc');
    }
  };

  const filtered = useMemo(() => {
    let result = [...mockStrategies];

    // Text search
    if (search) {
      const q = search.toLowerCase();
      result = result.filter(
        (s) => s.name.toLowerCase().includes(q) || s.symbol.toLowerCase().includes(q)
      );
    }

    // Status filter
    if (statusFilter !== 'All') {
      result = result.filter((s) => s.status === statusFilter);
    }

    // Sort
    result.sort((a, b) => {
      let cmp = 0;
      switch (sortKey) {
        case 'name': cmp = a.name.localeCompare(b.name); break;
        case 'symbol': cmp = a.symbol.localeCompare(b.symbol); break;
        case 'timeframe': cmp = a.timeframe.localeCompare(b.timeframe); break;
        case 'status': cmp = a.status.localeCompare(b.status); break;
        case 'winRate': cmp = a.winRate - b.winRate; break;
        case 'pf': cmp = a.pf - b.pf; break;
        case 'dailyR': cmp = a.dailyR - b.dailyR; break;
        case 'trades': cmp = a.trades - b.trades; break;
      }
      return sortDir === 'asc' ? cmp : -cmp;
    });

    return result;
  }, [search, statusFilter, sortKey, sortDir]);

  const deleteStrategy = filtered.find((s) => s.id === deleteId);

  const sortIcon = (key: SortKey) => {
    if (sortKey !== key) return '';
    return sortDir === 'asc' ? ' ^' : ' v';
  };

  const thStyle: React.CSSProperties = {
    color: 'var(--text-muted)',
    cursor: 'pointer',
    userSelect: 'none',
    fontWeight: 500,
    fontSize: '0.75rem',
    textAlign: 'left',
    padding: '8px 12px',
    whiteSpace: 'nowrap',
  };

  const tdStyle: React.CSSProperties = {
    padding: '10px 12px',
    fontSize: '0.875rem',
    whiteSpace: 'nowrap',
  };

  const inputStyle: React.CSSProperties = {
    background: 'var(--bg-input)',
    border: '1px solid var(--border)',
    color: 'var(--text-primary)',
    padding: '6px 12px',
    borderRadius: '8px',
    fontSize: '0.875rem',
  };

  return (
    <div>
      <PageHeader
        title="My Strategies"
        actions={
          <>
            <button
              className="px-3 py-1.5 rounded-lg text-sm"
              style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)', cursor: 'pointer' }}
            >
              Update Data
            </button>
            <button
              className="px-3 py-1.5 rounded-lg text-sm font-medium"
              style={{ background: 'var(--accent)', border: 'none', color: 'white', cursor: 'pointer' }}
            >
              + New Strategy
            </button>
          </>
        }
      />

      {/* Quick filter bar */}
      <div className="flex gap-3 mb-4">
        <input
          type="text"
          placeholder="Search strategies..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="flex-1"
          style={{ ...inputStyle, maxWidth: '320px' }}
        />
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          style={inputStyle}
        >
          <option value="All">All Statuses</option>
          <option value="On Track">On Track</option>
          <option value="Outperforming">Outperforming</option>
          <option value="Underperforming">Underperforming</option>
          <option value="Insufficient Data">Insufficient Data</option>
        </select>
        <span className="self-center text-xs" style={{ color: 'var(--text-muted)' }}>
          {filtered.length} of {mockStrategies.length}
        </span>
      </div>

      {/* Table */}
      <Card className="overflow-x-auto !p-0">
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              <th style={thStyle} onClick={() => handleSort('name')}>Name{sortIcon('name')}</th>
              <th style={thStyle} onClick={() => handleSort('symbol')}>Symbol{sortIcon('symbol')}</th>
              <th style={thStyle} onClick={() => handleSort('timeframe')}>TF{sortIcon('timeframe')}</th>
              <th style={thStyle} onClick={() => handleSort('status')}>Status{sortIcon('status')}</th>
              <th style={{ ...thStyle, textAlign: 'right' }} onClick={() => handleSort('winRate')}>WR{sortIcon('winRate')}</th>
              <th style={{ ...thStyle, textAlign: 'right' }} onClick={() => handleSort('pf')}>PF{sortIcon('pf')}</th>
              <th style={{ ...thStyle, textAlign: 'right' }} onClick={() => handleSort('dailyR')}>Daily R{sortIcon('dailyR')}</th>
              <th style={{ ...thStyle, textAlign: 'right' }} onClick={() => handleSort('trades')}>Trades{sortIcon('trades')}</th>
              <th style={thStyle}>Fwd Status</th>
              <th style={{ ...thStyle, textAlign: 'center' }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={10} style={{ padding: '40px', textAlign: 'center' }}>
                  <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                    {mockStrategies.length === 0
                      ? 'No strategies yet. Create your first one.'
                      : 'No strategies match the current filters.'}
                  </p>
                </td>
              </tr>
            ) : (
              filtered.map((strat) => (
                <tr
                  key={strat.id}
                  style={{ borderBottom: '1px solid var(--border)' }}
                  className="hover:bg-[var(--bg-input)]"
                >
                  <td style={tdStyle}>
                    <Link
                      href={`/strategies/${strat.id}`}
                      className="font-medium hover:underline"
                      style={{ color: 'var(--text-primary)' }}
                    >
                      {strat.name}
                    </Link>
                  </td>
                  <td style={tdStyle}>
                    <span className="font-medium">{strat.symbol}</span>
                  </td>
                  <td style={{ ...tdStyle, color: 'var(--text-secondary)' }}>{strat.timeframe}</td>
                  <td style={tdStyle}>
                    <span className="flex items-center gap-1.5">
                      <span
                        className="w-2 h-2 rounded-full inline-block flex-shrink-0"
                        style={{ background: statusDotColor[strat.status] }}
                      />
                      <span className="text-xs" style={{ color: statusDotColor[strat.status] }}>
                        {strat.status}
                      </span>
                    </span>
                  </td>
                  <td style={{ ...tdStyle, textAlign: 'right' }}>{strat.winRate.toFixed(1)}%</td>
                  <td style={{ ...tdStyle, textAlign: 'right' }}>{strat.pf.toFixed(2)}</td>
                  <td style={{ ...tdStyle, textAlign: 'right', color: strat.dailyR >= 1.5 ? 'var(--green)' : 'var(--text-primary)' }}>
                    {strat.dailyR >= 0 ? '+' : ''}{strat.dailyR.toFixed(2)}
                  </td>
                  <td style={{ ...tdStyle, textAlign: 'right' }}>{strat.trades}</td>
                  <td style={{ ...tdStyle, color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                    {strat.fwdStatus}
                  </td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>
                    <div className="flex items-center justify-center gap-1">
                      <Link
                        href={`/strategies/${strat.id}`}
                        className="w-7 h-7 rounded flex items-center justify-center text-xs"
                        style={{ color: 'var(--text-muted)', background: 'transparent' }}
                        title="View"
                      >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                          <circle cx="12" cy="12" r="3" />
                        </svg>
                      </Link>
                      <button
                        className="w-7 h-7 rounded flex items-center justify-center text-xs"
                        style={{ color: 'var(--text-muted)', background: 'transparent', border: 'none', cursor: 'pointer' }}
                        title="Edit"
                      >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                          <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                        </svg>
                      </button>
                      <button
                        className="w-7 h-7 rounded flex items-center justify-center text-xs"
                        style={{ color: 'var(--red)', background: 'transparent', border: 'none', cursor: 'pointer' }}
                        title="Delete"
                        onClick={() => setDeleteId(strat.id)}
                      >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <polyline points="3 6 5 6 21 6" />
                          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                        </svg>
                      </button>
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </Card>

      {/* Delete confirmation modal */}
      <Modal
        title="Delete Strategy"
        isOpen={deleteId !== null}
        onClose={() => setDeleteId(null)}
        width="420px"
      >
        {deleteStrategy && (
          <>
            <p className="text-sm mb-4" style={{ color: 'var(--text-secondary)' }}>
              Delete &apos;{deleteStrategy.name}&apos;? This cannot be undone.
            </p>
            <div className="flex gap-3 justify-end">
              <button
                className="px-4 py-2 rounded-lg text-sm"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)', cursor: 'pointer' }}
                onClick={() => setDeleteId(null)}
              >
                Cancel
              </button>
              <button
                className="px-4 py-2 rounded-lg text-sm font-medium"
                style={{ background: 'var(--red)', border: 'none', color: 'white', cursor: 'pointer' }}
                onClick={() => setDeleteId(null)}
              >
                Delete
              </button>
            </div>
          </>
        )}
      </Modal>
    </div>
  );
}
