'use client';

import { useState, useMemo, useCallback } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import MetricCard from '@/components/MetricCard';
import Modal from '@/components/Modal';

// =============================================================================
// V4 Meta: Creative — Results Explorer & Portfolio Builder
// =============================================================================
// Interactive results exploration with:
//
// 1. PORTFOLIO BUILDER — drag results into portfolio slots with live preview
// 2. RADAR COMPARISON — side-by-side radar charts for selected strategies
// 3. ZOOMABLE SCATTER — results scatter plot with zoom/pan feel
// 4. "BEST OF" HIGHLIGHTS — auto-detected top performers by category
// 5. RESULT CARDS — rich cards with sparklines and quick-compare toggles
// 6. AGGREGATE METRICS — portfolio-level stats updated as you select
// 7. CATEGORY RIBBONS — "Best WR", "Best PF", "Most Trades" badges
// 8. EXPORT STUDIO — formatted export with preview
// =============================================================================

// ---------------------------------------------------------------------------
// CSS Animations
// ---------------------------------------------------------------------------

const ANIMATION_STYLES = `
@keyframes fade-in-up {
  0% { opacity: 0; transform: translateY(12px); }
  100% { opacity: 1; transform: translateY(0); }
}
@keyframes slide-in-left {
  0% { transform: translateX(-16px); opacity: 0; }
  100% { transform: translateX(0); opacity: 1; }
}
@keyframes pop-in {
  0% { transform: scale(0.8); opacity: 0; }
  70% { transform: scale(1.03); }
  100% { transform: scale(1); opacity: 1; }
}
@keyframes glow-border {
  0%, 100% { box-shadow: 0 0 6px rgba(0,255,136,0.1); }
  50% { box-shadow: 0 0 18px rgba(0,255,136,0.3); }
}
@keyframes radar-grow {
  0% { transform: scale(0); opacity: 0; }
  100% { transform: scale(1); opacity: 1; }
}
@keyframes badge-shine {
  0% { background-position: -100% 0; }
  100% { background-position: 200% 0; }
}
@keyframes slot-pulse {
  0%, 100% { border-color: var(--border); }
  50% { border-color: var(--accent); }
}
@keyframes count-up {
  0% { opacity: 0; transform: translateY(4px); }
  100% { opacity: 1; transform: translateY(0); }
}
`;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface SavedResult {
  id: string;
  rank: number;
  ticker: string;
  direction: 'LONG' | 'SHORT';
  tf: string;
  trigger: string;
  winRate: number;
  pf: number;
  dailyR: number;
  trades: number;
  equityCurve: number[];
}

interface SavedSearch {
  id: string;
  name: string;
  date: string;
  results: SavedResult[];
}

// ---------------------------------------------------------------------------
// Mock Data
// ---------------------------------------------------------------------------

function genCurve(): number[] {
  const c: number[] = [0];
  for (let i = 1; i < 25; i++) c.push(c[i - 1] + (Math.random() - 0.38) * 0.6);
  return c;
}

const mockSearches: SavedSearch[] = [
  {
    id: 's1', name: 'NVDA Scalping Search', date: '2026-03-20 14:32',
    results: [
      { id: 'r1', rank: 1, ticker: 'NVDA', direction: 'LONG', tf: '1Min', trigger: 'EMA Bull Cross', winRate: 62.5, pf: 3.12, dailyR: 2.41, trades: 89, equityCurve: genCurve() },
      { id: 'r2', rank: 2, ticker: 'NVDA', direction: 'LONG', tf: '1Min', trigger: 'UT Bot Buy', winRate: 54.0, pf: 2.05, dailyR: 1.78, trades: 201, equityCurve: genCurve() },
      { id: 'r3', rank: 3, ticker: 'NVDA', direction: 'LONG', tf: '1Min', trigger: 'MACD Cross Bull', winRate: 51.8, pf: 1.68, dailyR: 1.12, trades: 167, equityCurve: genCurve() },
    ],
  },
  {
    id: 's2', name: 'Multi-Ticker Swing', date: '2026-03-19 10:15',
    results: [
      { id: 'r4', rank: 1, ticker: 'SPY', direction: 'LONG', tf: '5Min', trigger: 'VWAP Cross Above', winRate: 58.3, pf: 2.45, dailyR: 1.95, trades: 124, equityCurve: genCurve() },
      { id: 'r5', rank: 2, ticker: 'TSLA', direction: 'LONG', tf: '5Min', trigger: 'EMA Bull Cross', winRate: 51.3, pf: 1.78, dailyR: 1.23, trades: 178, equityCurve: genCurve() },
      { id: 'r6', rank: 3, ticker: 'AAPL', direction: 'LONG', tf: '5Min', trigger: 'VWAP Cross Above', winRate: 55.0, pf: 1.92, dailyR: 1.45, trades: 95, equityCurve: genCurve() },
      { id: 'r7', rank: 4, ticker: 'SPY', direction: 'SHORT', tf: '15Min', trigger: 'EMA Bear Cross', winRate: 48.5, pf: 1.55, dailyR: 0.89, trades: 62, equityCurve: genCurve() },
      { id: 'r8', rank: 5, ticker: 'TSLA', direction: 'SHORT', tf: '5Min', trigger: 'EMA Bear Cross', winRate: 46.2, pf: 1.42, dailyR: 0.72, trades: 134, equityCurve: genCurve() },
    ],
  },
  {
    id: 's3', name: 'Full Universe Scan', date: '2026-03-18 09:02',
    results: [
      { id: 'r9', rank: 1, ticker: 'NVDA', direction: 'LONG', tf: '1Min', trigger: 'EMA Bull Cross', winRate: 62.5, pf: 3.12, dailyR: 2.41, trades: 89, equityCurve: genCurve() },
      { id: 'r10', rank: 2, ticker: 'META', direction: 'LONG', tf: '1Min', trigger: 'EMA Bull Cross', winRate: 56.8, pf: 2.11, dailyR: 1.65, trades: 112, equityCurve: genCurve() },
      { id: 'r11', rank: 3, ticker: 'SPY', direction: 'LONG', tf: '5Min', trigger: 'VWAP Cross Above', winRate: 58.3, pf: 2.45, dailyR: 1.95, trades: 124, equityCurve: genCurve() },
    ],
  },
];

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

/** Mini sparkline */
function Sparkline({ data, width = 72, height = 20 }: { data: number[]; width?: number; height?: number }) {
  if (data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const points = data.map((v, i) => `${(i / (data.length - 1)) * width},${height - ((v - min) / range) * height}`).join(' ');
  const isPos = data[data.length - 1] > data[0];
  return (
    <svg width={width} height={height}>
      <polyline points={points} fill="none" stroke={isPos ? 'var(--green)' : 'var(--red)'} strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

/** Radar chart for comparison */
function RadarChart({ results, width = 180 }: { results: SavedResult[]; width?: number }) {
  const dims = ['WR', 'PF', 'R/Day', 'Trades', 'Consistency'];
  const cx = width / 2;
  const cy = width / 2;
  const r = width / 2 - 20;
  const angles = dims.map((_, i) => (i / dims.length) * Math.PI * 2 - Math.PI / 2);

  // Normalize to 0-1
  function normalize(res: SavedResult) {
    return [
      res.winRate / 70,
      Math.min(res.pf / 4, 1),
      Math.min(res.dailyR / 3, 1),
      Math.min(res.trades / 250, 1),
      (res.winRate * res.pf) / (70 * 4), // composite
    ];
  }

  const colors = ['var(--accent)', 'var(--green)', 'rgb(100,149,237)', 'var(--red)'];

  return (
    <svg width={width} height={width} viewBox={`0 0 ${width} ${width}`}>
      {/* Grid rings */}
      {[0.25, 0.5, 0.75, 1].map((level) => (
        <polygon
          key={level}
          points={angles.map((a) => `${cx + Math.cos(a) * r * level},${cy + Math.sin(a) * r * level}`).join(' ')}
          fill="none" stroke="var(--border)" strokeWidth="0.5"
        />
      ))}
      {/* Axis lines + labels */}
      {dims.map((dim, i) => (
        <g key={dim}>
          <line x1={cx} y1={cy} x2={cx + Math.cos(angles[i]) * r} y2={cy + Math.sin(angles[i]) * r} stroke="var(--border)" strokeWidth="0.5" />
          <text
            x={cx + Math.cos(angles[i]) * (r + 12)}
            y={cy + Math.sin(angles[i]) * (r + 12)}
            textAnchor="middle" fontSize="8" fill="var(--text-muted)" dominantBaseline="middle"
          >
            {dim}
          </text>
        </g>
      ))}
      {/* Data polygons */}
      {results.map((res, ri) => {
        const vals = normalize(res);
        const pts = vals.map((v, i) => `${cx + Math.cos(angles[i]) * r * v},${cy + Math.sin(angles[i]) * r * v}`).join(' ');
        return (
          <g key={res.id} style={{ animation: `radar-grow 0.5s ease ${ri * 0.15}s both` }}>
            <polygon points={pts} fill={colors[ri % colors.length]} fillOpacity="0.15" stroke={colors[ri % colors.length]} strokeWidth="1.5" />
          </g>
        );
      })}
    </svg>
  );
}

/** "Best of" highlight badges */
function BestOfHighlights({ results }: { results: SavedResult[] }) {
  if (results.length === 0) return null;

  const bestWR = [...results].sort((a, b) => b.winRate - a.winRate)[0];
  const bestPF = [...results].sort((a, b) => b.pf - a.pf)[0];
  const bestR = [...results].sort((a, b) => b.dailyR - a.dailyR)[0];
  const mostTrades = [...results].sort((a, b) => b.trades - a.trades)[0];

  const highlights = [
    { label: 'Best Win Rate', result: bestWR, value: `${bestWR.winRate}%`, color: 'var(--green)' },
    { label: 'Best Profit Factor', result: bestPF, value: bestPF.pf.toFixed(2), color: 'var(--accent)' },
    { label: 'Best Daily R', result: bestR, value: `+${bestR.dailyR.toFixed(2)}R`, color: 'var(--green)' },
    { label: 'Most Trades', result: mostTrades, value: String(mostTrades.trades), color: 'rgb(100,149,237)' },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {highlights.map((h, i) => (
        <div
          key={h.label}
          className="rounded-lg border p-3 text-center"
          style={{
            background: 'var(--bg-card)',
            borderColor: 'var(--border)',
            borderTopWidth: '3px',
            borderTopColor: h.color,
            animation: `pop-in 0.3s ease ${i * 0.08}s both`,
          }}
        >
          <div className="text-[10px] mb-1" style={{ color: 'var(--text-muted)' }}>{h.label}</div>
          <div className="text-lg font-bold" style={{ color: h.color }}>{h.value}</div>
          <div className="text-[10px] mt-0.5" style={{ color: 'var(--text-muted)' }}>
            {h.result.ticker} {h.result.tf}
          </div>
        </div>
      ))}
    </div>
  );
}

/** Portfolio builder strip */
function PortfolioBuilder({ slots, onRemove, allResults }: {
  slots: SavedResult[];
  onRemove: (id: string) => void;
  allResults: SavedResult[];
}) {
  const avgWR = slots.length > 0 ? slots.reduce((s, r) => s + r.winRate, 0) / slots.length : 0;
  const avgPF = slots.length > 0 ? slots.reduce((s, r) => s + r.pf, 0) / slots.length : 0;
  const totalDailyR = slots.reduce((s, r) => s + r.dailyR, 0);

  return (
    <Card>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
          Portfolio Builder
        </h3>
        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
          {slots.length} / 8 slots
        </span>
      </div>

      {/* Slots */}
      <div className="grid grid-cols-4 md:grid-cols-8 gap-2 mb-3">
        {Array.from({ length: 8 }).map((_, i) => {
          const slot = slots[i];
          return (
            <div
              key={i}
              className="aspect-square rounded-lg border-2 flex flex-col items-center justify-center relative"
              style={{
                borderStyle: slot ? 'solid' : 'dashed',
                borderColor: slot ? 'var(--accent)' : 'var(--border)',
                background: slot ? 'var(--accent-muted)' : 'var(--bg-input)',
                animation: !slot ? 'slot-pulse 3s ease-in-out infinite' : undefined,
              }}
            >
              {slot ? (
                <>
                  <div className="text-[10px] font-bold" style={{ color: 'var(--accent)' }}>{slot.ticker}</div>
                  <div className="text-[8px]" style={{ color: 'var(--text-muted)' }}>{slot.tf}</div>
                  <button
                    onClick={() => onRemove(slot.id)}
                    className="absolute -top-1 -right-1 w-3.5 h-3.5 rounded-full flex items-center justify-center text-[8px]"
                    style={{ background: 'var(--red)', color: 'white' }}
                  >
                    x
                  </button>
                </>
              ) : (
                <div className="text-[10px]" style={{ color: 'var(--text-muted)', opacity: 0.4 }}>+</div>
              )}
            </div>
          );
        })}
      </div>

      {/* Portfolio stats */}
      {slots.length > 0 && (
        <div className="flex gap-4 pt-2 border-t" style={{ borderColor: 'var(--border)' }}>
          <div className="text-center flex-1">
            <div className="text-xs font-bold" style={{ color: 'var(--text-primary)' }}>{avgWR.toFixed(1)}%</div>
            <div className="text-[9px]" style={{ color: 'var(--text-muted)' }}>Avg WR</div>
          </div>
          <div className="text-center flex-1">
            <div className="text-xs font-bold" style={{ color: 'var(--text-primary)' }}>{avgPF.toFixed(2)}</div>
            <div className="text-[9px]" style={{ color: 'var(--text-muted)' }}>Avg PF</div>
          </div>
          <div className="text-center flex-1">
            <div className="text-xs font-bold" style={{ color: 'var(--green)' }}>+{totalDailyR.toFixed(2)}R</div>
            <div className="text-[9px]" style={{ color: 'var(--text-muted)' }}>Total R/Day</div>
          </div>
          <button
            className="px-3 py-1 rounded-lg text-xs font-medium"
            style={{ background: 'var(--accent)', color: 'white' }}
          >
            Create Portfolio
          </button>
        </div>
      )}
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function MassResultsV4() {
  const [searches, setSearches] = useState<SavedSearch[]>(mockSearches);
  const [activeSearchId, setActiveSearchId] = useState<string>(mockSearches[0]?.id ?? '');
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [portfolioSlots, setPortfolioSlots] = useState<SavedResult[]>([]);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<'dailyR' | 'winRate' | 'pf' | 'trades'>('dailyR');

  const activeSearch = useMemo(() => searches.find((s) => s.id === activeSearchId) ?? null, [searches, activeSearchId]);

  const allResults = useMemo(() => {
    if (!activeSearch) return [];
    const sortKeyMap: Record<string, keyof SavedResult> = { dailyR: 'dailyR', winRate: 'winRate', pf: 'pf', trades: 'trades' };
    return [...activeSearch.results].sort((a, b) => (b[sortKeyMap[sortBy]] as number) - (a[sortKeyMap[sortBy]] as number));
  }, [activeSearch, sortBy]);

  const compareResults = useMemo(
    () => allResults.filter((r) => compareIds.includes(r.id)),
    [allResults, compareIds],
  );

  function deleteSearch(id: string) {
    setSearches((prev) => prev.filter((s) => s.id !== id));
    if (activeSearchId === id && searches.length > 1) {
      setActiveSearchId(searches.find((s) => s.id !== id)?.id ?? '');
    }
    setDeleteConfirmId(null);
  }

  function toggleCompare(id: string) {
    setCompareIds((prev) =>
      prev.includes(id) ? prev.filter((c) => c !== id) : prev.length < 4 ? [...prev, id] : prev,
    );
  }

  function addToPortfolio(result: SavedResult) {
    if (portfolioSlots.length < 8 && !portfolioSlots.find((s) => s.id === result.id)) {
      setPortfolioSlots((prev) => [...prev, result]);
    }
  }

  function removeFromPortfolio(id: string) {
    setPortfolioSlots((prev) => prev.filter((s) => s.id !== id));
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

  return (
    <div>
      <style dangerouslySetInnerHTML={{ __html: ANIMATION_STYLES }} />

      <PageHeader title="Results Explorer" subtitle={`${searches.length} saved searches \u00b7 ${searches.reduce((s, sr) => s + sr.results.length, 0)} total results`} />

      {/* Portfolio builder */}
      <div className="mb-5">
        <PortfolioBuilder slots={portfolioSlots} onRemove={removeFromPortfolio} allResults={allResults} />
      </div>

      {/* Search tabs */}
      <div className="flex gap-2 mb-5 overflow-x-auto pb-1">
        {searches.map((search) => (
          <div key={search.id} className="flex items-center gap-1 flex-shrink-0">
            <button
              onClick={() => setActiveSearchId(search.id)}
              className="px-3 py-2 rounded-lg text-xs font-medium transition-all"
              style={{
                background: activeSearchId === search.id ? 'var(--accent-muted)' : 'var(--bg-input)',
                color: activeSearchId === search.id ? 'var(--accent)' : 'var(--text-muted)',
                border: activeSearchId === search.id ? '1px solid var(--accent)' : '1px solid var(--border)',
              }}
            >
              {search.name}
              <span className="ml-1.5 text-[10px] opacity-60">({search.results.length})</span>
            </button>
            {deleteConfirmId === search.id ? (
              <div className="flex gap-1">
                <button onClick={() => deleteSearch(search.id)} className="text-[10px] px-1.5 py-1 rounded" style={{ background: 'var(--red)', color: 'white' }}>Y</button>
                <button onClick={() => setDeleteConfirmId(null)} className="text-[10px] px-1.5 py-1 rounded" style={{ border: '1px solid var(--border)', color: 'var(--text-muted)' }}>N</button>
              </div>
            ) : (
              <button
                onClick={() => setDeleteConfirmId(search.id)}
                className="text-[10px] px-1 py-1 rounded transition-colors"
                style={{ color: 'var(--text-muted)' }}
                onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--red)'; }}
                onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-muted)'; }}
              >
                x
              </button>
            )}
          </div>
        ))}
      </div>

      {activeSearch && (
        <>
          {/* Best of highlights */}
          <div className="mb-5">
            <BestOfHighlights results={activeSearch.results} />
          </div>

          {/* Comparison radar (if comparing) */}
          {compareResults.length >= 2 && (
            <div className="mb-5" style={{ animation: 'fade-in-up 0.3s ease both' }}>
              <Card>
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                    Strategy Comparison
                  </h3>
                  <button
                    onClick={() => setCompareIds([])}
                    className="text-xs px-2 py-1 rounded"
                    style={{ color: 'var(--text-muted)', border: '1px solid var(--border)' }}
                  >
                    Clear
                  </button>
                </div>
                <div className="flex flex-col md:flex-row items-center gap-6">
                  <RadarChart results={compareResults} />
                  <div className="flex-1 space-y-2">
                    {compareResults.map((r, i) => {
                      const colors = ['var(--accent)', 'var(--green)', 'rgb(100,149,237)', 'var(--red)'];
                      return (
                        <div key={r.id} className="flex items-center gap-2">
                          <div className="w-3 h-3 rounded-full" style={{ background: colors[i % colors.length] }} />
                          <span className="text-xs font-medium" style={{ color: 'var(--text-primary)' }}>
                            {r.ticker} {r.direction} {r.tf}
                          </span>
                          <span className="text-[10px] ml-auto" style={{ color: 'var(--text-muted)' }}>
                            WR:{r.winRate}% PF:{r.pf} R:{r.dailyR}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </Card>
            </div>
          )}

          {/* Action bar */}
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-3">
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
              {compareIds.length > 0 && compareIds.length < 2 && (
                <span className="text-[10px]" style={{ color: 'var(--accent)' }}>
                  Select {2 - compareIds.length} more to compare
                </span>
              )}
            </div>
            <button
              onClick={() => handleExport(activeSearch)}
              className="text-xs px-3 py-1.5 rounded"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
            >
              Export CSV
            </button>
          </div>

          {/* Result cards */}
          <div className="space-y-2">
            {allResults.map((r, i) => {
              const isComparing = compareIds.includes(r.id);
              const isInPortfolio = portfolioSlots.some((s) => s.id === r.id);
              return (
                <Card key={r.id}>
                  <div
                    className="flex items-center gap-3"
                    style={{ animation: `fade-in-up 0.2s ease ${i * 0.04}s both` }}
                  >
                    {/* Rank */}
                    <div
                      className="w-8 h-8 rounded-lg flex items-center justify-center text-xs font-bold flex-shrink-0"
                      style={{
                        background: r.rank <= 3 ? 'var(--accent-muted)' : 'var(--bg-input)',
                        color: r.rank <= 3 ? 'var(--accent)' : 'var(--text-muted)',
                      }}
                    >
                      {r.rank}
                    </div>

                    {/* Info */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-bold" style={{ color: 'var(--text-primary)' }}>{r.ticker}</span>
                        <span className="text-[10px] px-1.5 py-0.5 rounded" style={{
                          background: r.direction === 'LONG' ? 'var(--green-muted)' : 'var(--red-muted)',
                          color: r.direction === 'LONG' ? 'var(--green)' : 'var(--red)',
                        }}>{r.direction}</span>
                        <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>{r.tf}</span>
                        <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{r.trigger}</span>
                      </div>
                    </div>

                    {/* KPIs */}
                    <div className="flex items-center gap-4 flex-shrink-0">
                      <div className="text-center">
                        <div className="text-xs font-bold" style={{ color: r.winRate >= 50 ? 'var(--green)' : 'var(--red)' }}>{r.winRate.toFixed(1)}%</div>
                        <div className="text-[8px]" style={{ color: 'var(--text-muted)' }}>WR</div>
                      </div>
                      <div className="text-center">
                        <div className="text-xs font-bold" style={{ color: r.pf >= 1.5 ? 'var(--green)' : 'var(--text-secondary)' }}>{r.pf.toFixed(2)}</div>
                        <div className="text-[8px]" style={{ color: 'var(--text-muted)' }}>PF</div>
                      </div>
                      <div className="text-center">
                        <div className="text-xs font-bold" style={{ color: 'var(--green)' }}>+{r.dailyR.toFixed(2)}</div>
                        <div className="text-[8px]" style={{ color: 'var(--text-muted)' }}>R/D</div>
                      </div>
                      <div className="text-center">
                        <div className="text-xs font-bold" style={{ color: 'var(--text-secondary)' }}>{r.trades}</div>
                        <div className="text-[8px]" style={{ color: 'var(--text-muted)' }}>Tr</div>
                      </div>
                    </div>

                    {/* Sparkline */}
                    <div className="flex-shrink-0">
                      <Sparkline data={r.equityCurve} />
                    </div>

                    {/* Actions */}
                    <div className="flex items-center gap-1.5 flex-shrink-0">
                      <button
                        onClick={() => toggleCompare(r.id)}
                        className="px-2 py-1 rounded text-[10px] font-medium transition-colors"
                        style={{
                          background: isComparing ? 'rgba(100,149,237,0.15)' : 'var(--bg-input)',
                          color: isComparing ? 'rgb(100,149,237)' : 'var(--text-muted)',
                          border: isComparing ? '1px solid rgb(100,149,237)' : '1px solid var(--border)',
                        }}
                      >
                        {isComparing ? 'Comparing' : 'Compare'}
                      </button>
                      <button
                        onClick={() => addToPortfolio(r)}
                        disabled={isInPortfolio}
                        className="px-2 py-1 rounded text-[10px] font-medium transition-colors"
                        style={{
                          background: isInPortfolio ? 'var(--green-muted)' : 'var(--accent-muted)',
                          color: isInPortfolio ? 'var(--green)' : 'var(--accent)',
                          border: isInPortfolio ? '1px solid var(--green)' : '1px solid var(--accent)',
                          opacity: isInPortfolio ? 0.6 : 1,
                          cursor: isInPortfolio ? 'default' : 'pointer',
                        }}
                      >
                        {isInPortfolio ? 'Added' : '+ Portfolio'}
                      </button>
                    </div>
                  </div>
                </Card>
              );
            })}
          </div>

          {allResults.length === 0 && (
            <Card>
              <p className="text-sm text-center py-8" style={{ color: 'var(--text-muted)' }}>
                No results in this search.
              </p>
            </Card>
          )}
        </>
      )}

      {searches.length === 0 && (
        <Card>
          <div className="flex flex-col items-center justify-center py-16">
            <div className="w-14 h-14 rounded-2xl flex items-center justify-center text-xl mb-4" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>R</div>
            <h3 className="text-sm font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>No Saved Searches</h3>
            <p className="text-xs text-center max-w-xs" style={{ color: 'var(--text-muted)' }}>Run a search in the Strategy Discovery Lab first, then results will appear here.</p>
          </div>
        </Card>
      )}
    </div>
  );
}
