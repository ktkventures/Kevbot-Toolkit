'use client';

import { useState, useMemo } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

/* ========================================================================= */
/* TYPES                                                                      */
/* ========================================================================= */

interface SavedResult {
  rank: number;
  ticker: string;
  direction: 'LONG' | 'SHORT';
  tf: string;
  trigger: string;
  winRate: number;
  pf: number;
  dailyR: number;
  trades: number;
}

interface SavedSearch {
  id: string;
  name: string;
  date: string;
  resultCount: number;
  results: SavedResult[];
}

/* ========================================================================= */
/* MOCK DATA                                                                  */
/* ========================================================================= */

const mockSearches: SavedSearch[] = [
  {
    id: 's1', name: 'NVDA Scalping Search', date: '2026-03-20 14:32', resultCount: 3,
    results: [
      { rank: 1, ticker: 'NVDA', direction: 'LONG', tf: '1Min', trigger: 'EMA Bull Cross', winRate: 62.5, pf: 3.12, dailyR: 2.41, trades: 89 },
      { rank: 2, ticker: 'NVDA', direction: 'LONG', tf: '1Min', trigger: 'UT Bot Buy', winRate: 54.0, pf: 2.05, dailyR: 1.78, trades: 201 },
      { rank: 3, ticker: 'NVDA', direction: 'LONG', tf: '1Min', trigger: 'MACD Cross Bull', winRate: 51.8, pf: 1.68, dailyR: 1.12, trades: 167 },
    ],
  },
  {
    id: 's2', name: 'Multi-Ticker Swing', date: '2026-03-19 10:15', resultCount: 5,
    results: [
      { rank: 1, ticker: 'SPY', direction: 'LONG', tf: '5Min', trigger: 'VWAP Cross Above', winRate: 58.3, pf: 2.45, dailyR: 1.95, trades: 124 },
      { rank: 2, ticker: 'TSLA', direction: 'LONG', tf: '5Min', trigger: 'EMA Bull Cross', winRate: 51.3, pf: 1.78, dailyR: 1.23, trades: 178 },
      { rank: 3, ticker: 'AAPL', direction: 'LONG', tf: '5Min', trigger: 'VWAP Cross Above', winRate: 55.0, pf: 1.92, dailyR: 1.45, trades: 95 },
      { rank: 4, ticker: 'SPY', direction: 'SHORT', tf: '15Min', trigger: 'EMA Bear Cross', winRate: 48.5, pf: 1.55, dailyR: 0.89, trades: 62 },
      { rank: 5, ticker: 'TSLA', direction: 'SHORT', tf: '5Min', trigger: 'EMA Bear Cross', winRate: 46.2, pf: 1.42, dailyR: 0.72, trades: 134 },
    ],
  },
  {
    id: 's3', name: 'Full Universe Scan', date: '2026-03-18 09:02', resultCount: 3,
    results: [
      { rank: 1, ticker: 'NVDA', direction: 'LONG', tf: '1Min', trigger: 'EMA Bull Cross', winRate: 62.5, pf: 3.12, dailyR: 2.41, trades: 89 },
      { rank: 2, ticker: 'META', direction: 'LONG', tf: '1Min', trigger: 'EMA Bull Cross', winRate: 56.8, pf: 2.11, dailyR: 1.65, trades: 112 },
      { rank: 3, ticker: 'SPY', direction: 'LONG', tf: '5Min', trigger: 'VWAP Cross Above', winRate: 58.3, pf: 2.45, dailyR: 1.95, trades: 124 },
    ],
  },
];

/* ========================================================================= */
/* COMPONENT                                                                  */
/* ========================================================================= */

export default function MassResultsV3() {
  const [searches, setSearches] = useState<SavedSearch[]>(mockSearches);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<'dailyR' | 'winRate' | 'pf' | 'trades'>('dailyR');

  function deleteSearch(id: string) {
    setSearches((prev) => prev.filter((s) => s.id !== id));
    if (expandedId === id) setExpandedId(null);
    setDeleteConfirmId(null);
  }

  function handleExport(search: SavedSearch) {
    const csv = [
      'Rank,Ticker,Direction,Timeframe,Trigger,WinRate,PF,DailyR,Trades',
      ...search.results.map((r) =>
        `${r.rank},${r.ticker},${r.direction},${r.tf},${r.trigger},${r.winRate},${r.pf},${r.dailyR},${r.trades}`
      ),
    ].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${search.name.toLowerCase().replace(/\s+/g, '_')}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const expandedSearch = useMemo(
    () => searches.find((s) => s.id === expandedId) ?? null,
    [searches, expandedId]
  );

  const sortedResults = useMemo(() => {
    if (!expandedSearch) return [];
    return [...expandedSearch.results].sort((a, b) => (b[sortBy] as number) - (a[sortBy] as number));
  }, [expandedSearch, sortBy]);

  return (
    <div>
      <PageHeader title="Mass Results" subtitle="Saved mass search results" />

      <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
        {searches.length} saved search{searches.length !== 1 ? 'es' : ''}
      </p>

      <div className="space-y-3">
        {searches.map((search) => {
          const isExpanded = expandedId === search.id;

          return (
            <Card key={search.id}>
              {/* Search row */}
              <div
                className="flex items-center justify-between cursor-pointer"
                onClick={() => setExpandedId(isExpanded ? null : search.id)}
              >
                <div className="flex items-center gap-3 flex-1 min-w-0">
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    {isExpanded ? '▼' : '▶'}
                  </span>
                  <div className="min-w-0">
                    <h3 className="text-sm font-semibold truncate" style={{ color: 'var(--text-primary)' }}>
                      {search.name}
                    </h3>
                    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                      {search.date} &middot; {search.resultCount} result{search.resultCount !== 1 ? 's' : ''}
                    </p>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex items-center gap-2 shrink-0" onClick={(e) => e.stopPropagation()}>
                  {deleteConfirmId === search.id ? (
                    <div className="flex items-center gap-1.5">
                      <span className="text-xs" style={{ color: 'var(--red)' }}>Delete?</span>
                      <button
                        onClick={() => deleteSearch(search.id)}
                        className="text-xs px-2 py-1 rounded font-medium"
                        style={{ background: 'var(--red)', color: 'white' }}
                      >
                        Yes
                      </button>
                      <button
                        onClick={() => setDeleteConfirmId(null)}
                        className="text-xs px-2 py-1 rounded"
                        style={{ border: '1px solid var(--border)', color: 'var(--text-muted)' }}
                      >
                        No
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => setDeleteConfirmId(search.id)}
                      className="text-xs px-2 py-1.5 rounded transition-colors"
                      style={{ color: 'var(--text-muted)' }}
                      onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--red)'; e.currentTarget.style.background = 'var(--red-muted)'; }}
                      onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-muted)'; e.currentTarget.style.background = 'transparent'; }}
                    >
                      Delete
                    </button>
                  )}
                </div>
              </div>

              {/* Expanded: results table */}
              {isExpanded && expandedSearch && (
                <div className="mt-4 pt-3 border-t" style={{ borderColor: 'var(--border)' }}>
                  {/* Action bar */}
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <label className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Sort:</label>
                      <select
                        value={sortBy}
                        onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
                        className="px-2 py-1 rounded text-xs"
                        style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                      >
                        <option value="dailyR">Daily R</option>
                        <option value="winRate">Win Rate</option>
                        <option value="pf">Profit Factor</option>
                        <option value="trades">Trades</option>
                      </select>
                    </div>
                    <div className="flex gap-2">
                      <button
                        className="text-xs px-3 py-1.5 rounded font-medium"
                        style={{ background: 'var(--accent)', color: 'white' }}
                      >
                        Save All
                      </button>
                      <button
                        onClick={() => handleExport(expandedSearch)}
                        className="text-xs px-3 py-1.5 rounded"
                        style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
                      >
                        Export
                      </button>
                    </div>
                  </div>

                  {/* Table header */}
                  <div
                    className="grid items-center gap-2 py-2 px-2 text-[10px] font-medium border-b"
                    style={{ gridTemplateColumns: '32px 56px 50px 52px 1fr 56px 56px 64px 52px', borderColor: 'var(--border)', color: 'var(--text-muted)' }}
                  >
                    <span>#</span>
                    <span>Ticker</span>
                    <span>Dir</span>
                    <span>TF</span>
                    <span>Trigger</span>
                    <span className="text-right">WR%</span>
                    <span className="text-right">PF</span>
                    <span className="text-right">Daily R</span>
                    <span className="text-right">Trades</span>
                  </div>

                  {/* Table rows */}
                  <div>
                    {sortedResults.map((r) => (
                      <div
                        key={`${search.id}-${r.rank}`}
                        className="grid items-center gap-2 py-2 px-2 text-xs border-b transition-colors"
                        style={{ gridTemplateColumns: '32px 56px 50px 52px 1fr 56px 56px 64px 52px', borderColor: 'var(--border)' }}
                        onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-input)')}
                        onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                      >
                        <span style={{ color: 'var(--text-muted)' }}>{r.rank}</span>
                        <span className="font-medium" style={{ color: 'var(--text-primary)' }}>{r.ticker}</span>
                        <span
                          className="text-[10px] px-1 py-0.5 rounded text-center"
                          style={{
                            background: r.direction === 'LONG' ? 'var(--green-muted)' : 'var(--red-muted)',
                            color: r.direction === 'LONG' ? 'var(--green)' : 'var(--red)',
                          }}
                        >
                          {r.direction}
                        </span>
                        <span style={{ color: 'var(--text-secondary)' }}>{r.tf}</span>
                        <span className="truncate" style={{ color: 'var(--text-secondary)' }}>{r.trigger}</span>
                        <span className="text-right font-medium" style={{ color: r.winRate >= 50 ? 'var(--green)' : 'var(--red)' }}>
                          {r.winRate.toFixed(1)}
                        </span>
                        <span className="text-right font-medium" style={{ color: r.pf >= 1.5 ? 'var(--green)' : 'var(--text-secondary)' }}>
                          {r.pf.toFixed(2)}
                        </span>
                        <span className="text-right font-medium" style={{ color: r.dailyR > 0 ? 'var(--green)' : 'var(--red)' }}>
                          +{r.dailyR.toFixed(2)}
                        </span>
                        <span className="text-right" style={{ color: 'var(--text-secondary)' }}>{r.trades}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </Card>
          );
        })}

        {searches.length === 0 && (
          <Card>
            <p className="text-sm text-center py-8" style={{ color: 'var(--text-muted)' }}>
              No saved searches yet. Run a search in Mass Builder first.
            </p>
          </Card>
        )}
      </div>
    </div>
  );
}
