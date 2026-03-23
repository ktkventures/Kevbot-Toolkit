'use client';

import { useState, useMemo } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import MetricCard from '@/components/MetricCard';
import Modal from '@/components/Modal';

/* ========================================================================
   Types
   ======================================================================== */

interface SavedResult {
  rank: number;
  ticker: string;
  direction: 'LONG' | 'SHORT';
  tf: string;
  trigger: string;
  confluence: string[];
  winRate: number;
  pf: number;
  dailyR: number;
  trades: number;
  maxDD: number;
  totalR: number;
  avgR: number;
  rSquared: number;
  equityCurve: number[];
  exitTrigger: string;
  stopDesc: string;
  targetDesc: string;
  selected: boolean;
}

interface SavedSearch {
  id: string;
  name: string;
  date: string;
  tickers: string[];
  timeframes: string[];
  directions: string[];
  session: string;
  results: SavedResult[];
  bestDailyR: number;
  bestWR: number;
  bestPF: number;
  status: 'completed' | 'pending';
}

/* ========================================================================
   Mock Data
   ======================================================================== */

function generateEquityCurve(totalR: number, trades: number): number[] {
  const curve: number[] = [0];
  const avgStep = totalR / trades;
  for (let i = 1; i <= trades; i++) {
    const prev = curve[i - 1];
    const noise = (Math.random() - 0.4) * Math.abs(avgStep) * 2;
    curve.push(prev + avgStep * 0.7 + noise);
  }
  const scale = totalR / (curve[curve.length - 1] || 1);
  return curve.map((v) => Math.round(v * scale * 100) / 100);
}

const mockSavedSearches: SavedSearch[] = [
  {
    id: 's1', name: 'NVDA Scalping Search', date: '2026-03-20 14:32',
    tickers: ['NVDA'], timeframes: ['1Min'], directions: ['LONG'], session: 'RTH',
    bestDailyR: 2.41, bestWR: 62.5, bestPF: 3.12, status: 'completed',
    results: [
      { rank: 1, ticker: 'NVDA', direction: 'LONG', tf: '1Min', trigger: '[C] EMA Bull Cross', confluence: ['SML', 'M>S+'], winRate: 62.5, pf: 3.12, dailyR: 2.41, trades: 89, maxDD: -1.8, totalR: 34.2, avgR: 0.38, rSquared: 0.87, equityCurve: generateEquityCurve(34.2, 89), exitTrigger: 'EMA Bear Cross', stopDesc: 'ATR x1.5', targetDesc: 'R:R 2.0', selected: false },
      { rank: 2, ticker: 'NVDA', direction: 'LONG', tf: '1Min', trigger: '[HM] UT Bot Buy', confluence: ['SML', 'M>S+', '>V'], winRate: 54.0, pf: 2.05, dailyR: 1.78, trades: 201, maxDD: -2.5, totalR: 25.1, avgR: 0.12, rSquared: 0.79, equityCurve: generateEquityCurve(25.1, 201), exitTrigger: 'MACD Cross Bear', stopDesc: 'ATR x1.5', targetDesc: 'R:R 3.0', selected: false },
      { rank: 3, ticker: 'NVDA', direction: 'LONG', tf: '1Min', trigger: '[C] MACD Cross Bull', confluence: ['SML'], winRate: 51.8, pf: 1.68, dailyR: 1.12, trades: 167, maxDD: -2.8, totalR: 15.3, avgR: 0.09, rSquared: 0.72, equityCurve: generateEquityCurve(15.3, 167), exitTrigger: 'MACD Cross Bear', stopDesc: 'Swing 5', targetDesc: 'None', selected: false },
    ],
  },
  {
    id: 's2', name: 'Multi-Ticker Swing', date: '2026-03-19 10:15',
    tickers: ['SPY', 'AAPL', 'TSLA'], timeframes: ['5Min', '15Min'], directions: ['LONG', 'SHORT'], session: 'RTH',
    bestDailyR: 1.95, bestWR: 58.3, bestPF: 2.45, status: 'completed',
    results: [
      { rank: 1, ticker: 'SPY', direction: 'LONG', tf: '5Min', trigger: '[C] VWAP Cross Above', confluence: ['SML'], winRate: 58.3, pf: 2.45, dailyR: 1.95, trades: 124, maxDD: -2.1, totalR: 28.5, avgR: 0.23, rSquared: 0.82, equityCurve: generateEquityCurve(28.5, 124), exitTrigger: 'VWAP Cross Below', stopDesc: 'Swing 5', targetDesc: 'None', selected: false },
      { rank: 2, ticker: 'TSLA', direction: 'LONG', tf: '5Min', trigger: '[C] EMA Bull Cross', confluence: ['SML'], winRate: 51.3, pf: 1.78, dailyR: 1.23, trades: 178, maxDD: -2.9, totalR: 18.7, avgR: 0.11, rSquared: 0.71, equityCurve: generateEquityCurve(18.7, 178), exitTrigger: 'EMA Bear Cross', stopDesc: 'Fixed $1.00', targetDesc: 'R:R 2.0', selected: false },
      { rank: 3, ticker: 'AAPL', direction: 'LONG', tf: '5Min', trigger: '[L0] VWAP Cross Above', confluence: ['M>S+'], winRate: 55.0, pf: 1.92, dailyR: 1.45, trades: 95, maxDD: -2.3, totalR: 14.9, avgR: 0.16, rSquared: 0.74, equityCurve: generateEquityCurve(14.9, 95), exitTrigger: 'VWAP Cross Below', stopDesc: 'ATR x2.0', targetDesc: 'ATR x3.0', selected: false },
      { rank: 4, ticker: 'SPY', direction: 'SHORT', tf: '15Min', trigger: '[C] EMA Bear Cross', confluence: ['LMS'], winRate: 48.5, pf: 1.55, dailyR: 0.89, trades: 62, maxDD: -3.2, totalR: 8.1, avgR: 0.13, rSquared: 0.63, equityCurve: generateEquityCurve(8.1, 62), exitTrigger: 'EMA Bull Cross', stopDesc: 'ATR x1.5', targetDesc: 'R:R 2.5', selected: false },
      { rank: 5, ticker: 'TSLA', direction: 'SHORT', tf: '5Min', trigger: '[C] EMA Bear Cross', confluence: ['LMS', 'M<S-'], winRate: 46.2, pf: 1.42, dailyR: 0.72, trades: 134, maxDD: -3.5, totalR: 6.5, avgR: 0.05, rSquared: 0.58, equityCurve: generateEquityCurve(6.5, 134), exitTrigger: 'EMA Bull Cross', stopDesc: 'Swing 5', targetDesc: 'None', selected: false },
    ],
  },
  {
    id: 's3', name: 'Full Universe Scan', date: '2026-03-18 09:02',
    tickers: ['NVDA', 'SPY', 'AAPL', 'TSLA', 'META'], timeframes: ['1Min', '5Min'], directions: ['LONG'], session: 'RTH',
    bestDailyR: 2.41, bestWR: 62.5, bestPF: 3.12, status: 'completed',
    results: [
      { rank: 1, ticker: 'NVDA', direction: 'LONG', tf: '1Min', trigger: '[C] EMA Bull Cross', confluence: ['SML', 'M>S+'], winRate: 62.5, pf: 3.12, dailyR: 2.41, trades: 89, maxDD: -1.8, totalR: 34.2, avgR: 0.38, rSquared: 0.87, equityCurve: generateEquityCurve(34.2, 89), exitTrigger: 'EMA Bear Cross', stopDesc: 'ATR x1.5', targetDesc: 'R:R 2.0', selected: false },
      { rank: 2, ticker: 'META', direction: 'LONG', tf: '1Min', trigger: '[C] EMA Bull Cross', confluence: ['SML'], winRate: 56.8, pf: 2.11, dailyR: 1.65, trades: 112, maxDD: -2.4, totalR: 22.1, avgR: 0.20, rSquared: 0.76, equityCurve: generateEquityCurve(22.1, 112), exitTrigger: 'EMA Bear Cross', stopDesc: 'ATR x1.5', targetDesc: 'R:R 2.0', selected: false },
      { rank: 3, ticker: 'SPY', direction: 'LONG', tf: '5Min', trigger: '[C] VWAP Cross Above', confluence: ['SML'], winRate: 58.3, pf: 2.45, dailyR: 1.95, trades: 124, maxDD: -2.1, totalR: 28.5, avgR: 0.23, rSquared: 0.82, equityCurve: generateEquityCurve(28.5, 124), exitTrigger: 'VWAP Cross Below', stopDesc: 'Swing 5', targetDesc: 'None', selected: false },
    ],
  },
];

/* ========================================================================
   Sub-Components
   ======================================================================== */

function MiniEquityCurve({ data, height = 60 }: { data: number[]; height?: number }) {
  if (data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const w = 160;
  const h = height;
  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - ((v - min) / range) * (h - 8) - 4;
    return `${x},${y}`;
  }).join(' ');
  const final = data[data.length - 1];
  const color = final >= 0 ? 'var(--green, #4CAF50)' : 'var(--red, #f44336)';
  const fillColor = final >= 0 ? 'rgba(76,175,80,0.12)' : 'rgba(244,67,54,0.12)';
  const fillPoints = `0,${h} ${points} ${w},${h}`;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} width="100%" height={height} preserveAspectRatio="none">
      <polygon points={fillPoints} fill={fillColor} />
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" />
    </svg>
  );
}

/* ========================================================================
   Main Component
   ======================================================================== */

export default function MassResultsV2() {
  const [searches, setSearches] = useState<SavedSearch[]>(mockSavedSearches);
  const [expandedSearchId, setExpandedSearchId] = useState<string | null>('s1');
  const [compareMode, setCompareMode] = useState(false);
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [showCompareModal, setShowCompareModal] = useState(false);
  const [showExportModal, setShowExportModal] = useState(false);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);

  // Per-search filters
  const [resultSort, setResultSort] = useState('Daily R');
  const [resultFilterTicker, setResultFilterTicker] = useState('');
  const [resultFilterMinWR, setResultFilterMinWR] = useState(0);

  // Selection state within expanded search
  const [selectedResultIds, setSelectedResultIds] = useState<Set<string>>(new Set());

  const expandedSearch = expandedSearchId
    ? searches.find((s) => s.id === expandedSearchId)
    : null;

  // Filter + sort results
  const filteredResults = useMemo(() => {
    if (!expandedSearch) return [];
    let results = [...expandedSearch.results];
    if (resultFilterTicker) {
      results = results.filter((r) => r.ticker === resultFilterTicker);
    }
    if (resultFilterMinWR > 0) {
      results = results.filter((r) => r.winRate >= resultFilterMinWR);
    }
    const sortMap: Record<string, keyof SavedResult> = {
      'Daily R': 'dailyR', 'Win Rate': 'winRate', 'Profit Factor': 'pf',
      'Total R': 'totalR', 'Trades': 'trades', 'R-Squared': 'rSquared',
    };
    const sk = sortMap[resultSort] || 'dailyR';
    results.sort((a, b) => (b[sk] as number) - (a[sk] as number));
    return results;
  }, [expandedSearch, resultSort, resultFilterTicker, resultFilterMinWR]);

  // Unique tickers for filter dropdown
  const availableTickers = useMemo(() => {
    if (!expandedSearch) return [];
    return Array.from(new Set(expandedSearch.results.map((r) => r.ticker))).sort();
  }, [expandedSearch]);

  function deleteSearch(id: string) {
    setSearches((prev) => prev.filter((s) => s.id !== id));
    if (expandedSearchId === id) setExpandedSearchId(null);
    setConfirmDeleteId(null);
  }

  function copySearch(id: string) {
    const source = searches.find((s) => s.id === id);
    if (!source) return;
    const copy: SavedSearch = {
      ...source,
      id: `s${Date.now()}`,
      name: `${source.name} (copy)`,
      results: [], // Copied search has no results yet
      status: 'pending',
    };
    setSearches((prev) => [...prev, copy]);
  }

  function toggleResultSelection(key: string) {
    setSelectedResultIds((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  function toggleCompareItem(searchId: string) {
    setCompareIds((prev) => {
      if (prev.includes(searchId)) return prev.filter((id) => id !== searchId);
      if (prev.length >= 3) return prev; // Max 3
      return [...prev, searchId];
    });
  }

  // Get compare strategies from selected result IDs
  const compareStrategies = useMemo(() => {
    if (!expandedSearch) return [];
    return expandedSearch.results.filter((_, idx) => selectedResultIds.has(`${expandedSearch.id}-${idx}`));
  }, [expandedSearch, selectedResultIds]);

  return (
    <div>
      <PageHeader title="Mass Results" subtitle="Browse and compare saved mass search results" />

      {/* Action bar */}
      <div className="flex items-center justify-between mb-5">
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
          {searches.length} saved search{searches.length !== 1 ? 'es' : ''}
        </p>
        <div className="flex gap-2">
          {selectedResultIds.size >= 2 && (
            <button
              onClick={() => setShowCompareModal(true)}
              className="px-4 py-2 rounded-lg text-xs font-medium"
              style={{ background: 'var(--accent-muted)', border: '1px solid var(--accent)', color: 'var(--accent)' }}
            >
              Compare Selected ({selectedResultIds.size})
            </button>
          )}
          {selectedResultIds.size > 0 && (
            <>
              <button
                className="px-4 py-2 rounded-lg text-xs font-medium"
                style={{ background: 'var(--accent)', color: 'white' }}
              >
                Save {selectedResultIds.size} to My Strategies
              </button>
              <button
                className="px-4 py-2 rounded-lg text-xs font-medium"
                style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
              >
                Create Portfolio
              </button>
              <button
                onClick={() => setShowExportModal(true)}
                className="px-4 py-2 rounded-lg text-xs font-medium"
                style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
              >
                Export CSV
              </button>
            </>
          )}
        </div>
      </div>

      {/* ====== Saved Searches List ====== */}
      <div className="space-y-3 mb-6">
        {searches.map((search) => {
          const isExpanded = expandedSearchId === search.id;
          const tickerStr = search.tickers.length <= 4
            ? search.tickers.join(', ')
            : `${search.tickers.slice(0, 4).join(', ')} +${search.tickers.length - 4}`;

          return (
            <Card key={search.id}>
              {/* Search header row */}
              <div className="flex items-center gap-4">
                <div className="flex-1 min-w-0 cursor-pointer" onClick={() => setExpandedSearchId(isExpanded ? null : search.id)}>
                  <div className="flex items-center gap-2 mb-0.5">
                    <h3 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                      {search.name}
                    </h3>
                    <span
                      className="text-[10px] px-1.5 py-0.5 rounded"
                      style={{
                        background: search.status === 'completed' ? 'rgba(76,175,80,0.15)' : 'rgba(245,158,11,0.15)',
                        color: search.status === 'completed' ? 'var(--green, #4CAF50)' : '#f59e0b',
                      }}
                    >
                      {search.status}
                    </span>
                  </div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    {search.results.length} results &middot; {tickerStr} &middot;
                    {' '}{search.timeframes.join(', ')} &middot; {search.date}
                  </p>
                </div>

                {/* Summary KPIs */}
                <div className="hidden md:flex gap-4 mr-4">
                  <div className="text-center">
                    <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Best Daily R</p>
                    <p className="text-sm font-bold" style={{ color: search.bestDailyR > 0 ? 'var(--green, #4CAF50)' : 'var(--text-primary)' }}>
                      +{search.bestDailyR.toFixed(2)}
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Best WR</p>
                    <p className="text-sm font-bold" style={{ color: 'var(--text-primary)' }}>
                      {search.bestWR.toFixed(1)}%
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Best PF</p>
                    <p className="text-sm font-bold" style={{ color: 'var(--text-primary)' }}>
                      {search.bestPF.toFixed(2)}
                    </p>
                  </div>
                </div>

                {/* Action buttons */}
                <div className="flex gap-1.5">
                  <button
                    onClick={() => setExpandedSearchId(isExpanded ? null : search.id)}
                    className="px-3 py-1.5 rounded-lg text-xs font-medium"
                    style={{
                      background: isExpanded ? 'var(--accent)' : 'var(--bg-input)',
                      color: isExpanded ? 'white' : 'var(--text-secondary)',
                      border: isExpanded ? 'none' : '1px solid var(--border)',
                    }}
                  >
                    {isExpanded ? 'Collapse' : 'View'}
                  </button>
                  <button
                    className="px-3 py-1.5 rounded-lg text-xs"
                    style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
                    title="Load into Mass Builder"
                  >
                    Load
                  </button>
                  <button
                    onClick={() => copySearch(search.id)}
                    className="px-3 py-1.5 rounded-lg text-xs"
                    style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
                  >
                    Copy
                  </button>
                  <button
                    onClick={() => setConfirmDeleteId(search.id)}
                    className="px-3 py-1.5 rounded-lg text-xs"
                    style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--red, #f44336)' }}
                  >
                    Delete
                  </button>
                </div>
              </div>

              {/* Delete confirmation */}
              {confirmDeleteId === search.id && (
                <div
                  className="mt-3 p-3 rounded-lg flex items-center justify-between"
                  style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)' }}
                >
                  <p className="text-xs" style={{ color: 'var(--red, #f44336)' }}>
                    Delete &quot;{search.name}&quot;? This cannot be undone.
                  </p>
                  <div className="flex gap-2">
                    <button
                      onClick={() => deleteSearch(search.id)}
                      className="px-3 py-1 rounded text-xs font-medium"
                      style={{ background: 'var(--red, #f44336)', color: 'white' }}
                    >
                      Yes, Delete
                    </button>
                    <button
                      onClick={() => setConfirmDeleteId(null)}
                      className="px-3 py-1 rounded text-xs"
                      style={{ border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}

              {/* Expanded: Results browser */}
              {isExpanded && search.results.length > 0 && (
                <div className="mt-4 border-t pt-4" style={{ borderColor: 'var(--border)' }}>
                  {/* Config summary */}
                  <div
                    className="rounded-lg p-3 mb-4 text-xs"
                    style={{ background: 'var(--bg-input)' }}
                  >
                    <p style={{ color: 'var(--text-muted)' }}>
                      <strong>Tickers:</strong> {search.tickers.join(', ')} &middot;
                      <strong> Timeframes:</strong> {search.timeframes.join(', ')} &middot;
                      <strong> Directions:</strong> {search.directions.join(', ')} &middot;
                      <strong> Session:</strong> {search.session}
                    </p>
                  </div>

                  {/* Filters row */}
                  <div className="flex flex-wrap gap-3 mb-4 items-end">
                    <div>
                      <label className="text-[10px] font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>Sort by</label>
                      <select
                        value={resultSort}
                        onChange={(e) => setResultSort(e.target.value)}
                        className="px-2 py-1.5 rounded-lg text-xs"
                        style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                      >
                        {['Daily R', 'Win Rate', 'Profit Factor', 'Total R', 'Trades', 'R-Squared'].map((o) => (
                          <option key={o}>{o}</option>
                        ))}
                      </select>
                    </div>
                    {availableTickers.length > 1 && (
                      <div>
                        <label className="text-[10px] font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>Ticker</label>
                        <select
                          value={resultFilterTicker}
                          onChange={(e) => setResultFilterTicker(e.target.value)}
                          className="px-2 py-1.5 rounded-lg text-xs"
                          style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                        >
                          <option value="">All</option>
                          {availableTickers.map((t) => <option key={t}>{t}</option>)}
                        </select>
                      </div>
                    )}
                    <div>
                      <label className="text-[10px] font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>Min WR%</label>
                      <input
                        type="number" min={0} max={100} step={5}
                        value={resultFilterMinWR}
                        onChange={(e) => setResultFilterMinWR(parseFloat(e.target.value) || 0)}
                        className="w-16 px-2 py-1.5 rounded-lg text-xs"
                        style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                      />
                    </div>
                    <p className="text-xs pb-1" style={{ color: 'var(--text-muted)' }}>
                      {filteredResults.length} of {search.results.length} results
                    </p>
                  </div>

                  {/* Result cards */}
                  <div className="space-y-2">
                    {filteredResults.map((result, idx) => {
                      const resultKey = `${search.id}-${idx}`;
                      const isSelected = selectedResultIds.has(resultKey);

                      return (
                        <div
                          key={idx}
                          className="rounded-lg border p-3 transition-colors"
                          style={{
                            background: isSelected ? 'var(--accent-muted)' : 'var(--bg-card)',
                            borderColor: isSelected ? 'var(--accent)' : 'var(--border)',
                          }}
                        >
                          <div className="flex gap-3 items-start">
                            {/* Checkbox */}
                            <div className="pt-0.5">
                              <input
                                type="checkbox"
                                checked={isSelected}
                                onChange={() => toggleResultSelection(resultKey)}
                                className="rounded"
                              />
                            </div>

                            {/* Info */}
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2 mb-0.5 flex-wrap">
                                <span
                                  className="text-[10px] font-bold px-1.5 py-0.5 rounded"
                                  style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
                                >
                                  #{result.rank}
                                </span>
                                <span className="text-sm font-bold" style={{ color: 'var(--text-primary)' }}>
                                  {result.ticker}
                                </span>
                                <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>{result.tf}</span>
                                <span
                                  className="text-xs font-medium"
                                  style={{ color: result.direction === 'LONG' ? 'var(--green, #4CAF50)' : 'var(--red, #f44336)' }}
                                >
                                  {result.direction}
                                </span>
                              </div>
                              <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                                {result.trigger}
                                {result.confluence.length > 0 && (
                                  <> + {result.confluence.map((c) => (
                                    <span
                                      key={c}
                                      className="inline-block mx-0.5 px-1 py-0 rounded font-mono text-[10px]"
                                      style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}
                                    >
                                      {c}
                                    </span>
                                  ))}</>
                                )}
                              </p>
                              <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                                Exit: {result.exitTrigger} &middot; Stop: {result.stopDesc} &middot; Target: {result.targetDesc}
                              </p>

                              {/* KPIs */}
                              <div className="grid grid-cols-4 md:grid-cols-8 gap-2 mt-2">
                                {[
                                  { label: 'Trades', value: result.trades.toString() },
                                  { label: 'WR', value: `${result.winRate.toFixed(1)}%` },
                                  { label: 'PF', value: result.pf.toFixed(2) },
                                  { label: 'Avg R', value: `${result.avgR >= 0 ? '+' : ''}${result.avgR.toFixed(2)}` },
                                  { label: 'Total R', value: `${result.totalR >= 0 ? '+' : ''}${result.totalR.toFixed(1)}` },
                                  { label: 'Daily R', value: `${result.dailyR >= 0 ? '+' : ''}${result.dailyR.toFixed(2)}` },
                                  { label: 'R\u00B2', value: result.rSquared.toFixed(2) },
                                  { label: 'Max DD', value: `${result.maxDD.toFixed(1)}R` },
                                ].map((kpi) => (
                                  <div key={kpi.label} className="text-center">
                                    <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{kpi.label}</p>
                                    <p className="text-xs font-bold" style={{ color: 'var(--text-primary)' }}>{kpi.value}</p>
                                  </div>
                                ))}
                              </div>
                            </div>

                            {/* Equity curve */}
                            <div className="w-36 flex-shrink-0">
                              <MiniEquityCurve data={result.equityCurve} height={60} />
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Expanded but empty */}
              {isExpanded && search.results.length === 0 && (
                <div className="mt-4 border-t pt-4 text-center py-6" style={{ borderColor: 'var(--border)' }}>
                  <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                    No results in this search. Load it into Mass Builder to run the analysis.
                  </p>
                </div>
              )}
            </Card>
          );
        })}
      </div>

      {/* Empty state */}
      {searches.length === 0 && (
        <Card className="mt-4">
          <div className="text-center py-12">
            <p className="text-sm mb-1" style={{ color: 'var(--text-secondary)' }}>
              No mass searches yet
            </p>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
              Go to Mass Builder to create your first search.
            </p>
          </div>
        </Card>
      )}

      {/* ====== Compare Modal ====== */}
      <Modal
        title="Compare Strategies"
        isOpen={showCompareModal}
        onClose={() => setShowCompareModal(false)}
        width="900px"
      >
        {compareStrategies.length >= 2 ? (
          <div>
            <p className="text-sm mb-4" style={{ color: 'var(--text-secondary)' }}>
              Side-by-side comparison of {compareStrategies.length} strategies
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border)' }}>
                    <th className="text-left py-2 px-2" style={{ color: 'var(--text-muted)' }}>Metric</th>
                    {compareStrategies.map((s, i) => (
                      <th key={i} className="text-center py-2 px-2" style={{ color: 'var(--text-primary)' }}>
                        {s.ticker} {s.tf} {s.direction}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[
                    { label: 'Entry', fn: (s: SavedResult) => s.trigger },
                    { label: 'Confluence', fn: (s: SavedResult) => s.confluence.join(', ') || '-' },
                    { label: 'Win Rate', fn: (s: SavedResult) => `${s.winRate.toFixed(1)}%` },
                    { label: 'Profit Factor', fn: (s: SavedResult) => s.pf.toFixed(2) },
                    { label: 'Daily R', fn: (s: SavedResult) => `+${s.dailyR.toFixed(2)}` },
                    { label: 'Total R', fn: (s: SavedResult) => `+${s.totalR.toFixed(1)}` },
                    { label: 'Trades', fn: (s: SavedResult) => s.trades.toString() },
                    { label: 'Max DD', fn: (s: SavedResult) => `${s.maxDD.toFixed(1)}R` },
                    { label: 'R-Squared', fn: (s: SavedResult) => s.rSquared.toFixed(2) },
                    { label: 'Stop', fn: (s: SavedResult) => s.stopDesc },
                    { label: 'Target', fn: (s: SavedResult) => s.targetDesc },
                  ].map((row) => (
                    <tr key={row.label} style={{ borderBottom: '1px solid var(--border)' }}>
                      <td className="py-2 px-2 font-medium" style={{ color: 'var(--text-muted)' }}>{row.label}</td>
                      {compareStrategies.map((s, i) => (
                        <td key={i} className="py-2 px-2 text-center" style={{ color: 'var(--text-secondary)' }}>
                          {row.fn(s)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {/* Equity curves side by side */}
            <div className="grid grid-cols-3 gap-4 mt-4">
              {compareStrategies.map((s, i) => (
                <div key={i}>
                  <p className="text-[10px] text-center mb-1" style={{ color: 'var(--text-muted)' }}>
                    {s.ticker} {s.tf} {s.direction}
                  </p>
                  <MiniEquityCurve data={s.equityCurve} height={80} />
                </div>
              ))}
            </div>
          </div>
        ) : (
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
            Select at least 2 strategies to compare.
          </p>
        )}
      </Modal>

      {/* ====== Export Modal ====== */}
      <Modal
        title="Export Results"
        isOpen={showExportModal}
        onClose={() => setShowExportModal(false)}
        width="480px"
      >
        <p className="text-sm mb-4" style={{ color: 'var(--text-secondary)' }}>
          Export {selectedResultIds.size} selected strateg{selectedResultIds.size !== 1 ? 'ies' : 'y'} as CSV.
        </p>
        <div
          className="rounded-lg p-3 mb-4 font-mono text-xs overflow-x-auto"
          style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}
        >
          <p>ticker,direction,tf,trigger,confluence,win_rate,pf,daily_r,trades,max_dd,total_r</p>
          {compareStrategies.slice(0, 3).map((s, i) => (
            <p key={i}>{s.ticker},{s.direction},{s.tf},&quot;{s.trigger}&quot;,&quot;{s.confluence.join(';')}&quot;,{s.winRate},{s.pf},{s.dailyR},{s.trades},{s.maxDD},{s.totalR}</p>
          ))}
          {compareStrategies.length > 3 && <p>...</p>}
        </div>
        <div className="flex gap-3">
          <button
            className="px-4 py-2 rounded-lg text-sm font-medium flex-1"
            style={{ background: 'var(--accent)', color: 'white' }}
          >
            Download CSV
          </button>
          <button
            className="px-4 py-2 rounded-lg text-sm font-medium flex-1"
            style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
          >
            Copy to Clipboard
          </button>
        </div>
      </Modal>
    </div>
  );
}
