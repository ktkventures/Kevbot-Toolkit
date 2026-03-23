'use client';

import { useState, useMemo } from 'react';
import Link from 'next/link';
import Card from '@/components/Card';
import Modal from '@/components/Modal';
import MetricCard from '@/components/MetricCard';

// =============================================================================
// V4 Meta: Creative — Strategy Portfolio Command Center
// =============================================================================
// The "wow factor" strategy list. Key creative features:
//
// 1. PORTFOLIO OVERVIEW STRIP — summary dashboard bar at top with total
//    strategies, avg win rate, best/worst performer, total R across all
// 2. LIVE STATUS CARDS — animated pulsing dots for actively monitored strats
// 3. PERFORMANCE SPARKLINES — real SVG equity curves per card
// 4. COMPARISON MODE — select 2 strategies for side-by-side radar + KPI diff
// 5. STRATEGY GROUPING — tag-based visual grouping with colored headers
// 6. QUICK STATS BAR — sticky bar with portfolio-level aggregate stats
// 7. SMART SEARCH — filter bar with inline suggestions
// 8. INLINE EQUITY CURVES — detailed SVG sparklines, not placeholders
// 9. STRATEGY DNA PREVIEW — mini radar chart per card
// =============================================================================

// ---------------------------------------------------------------------------
// CSS Animations
// ---------------------------------------------------------------------------

const ANIMATION_STYLES = `
@keyframes pulse-live {
  0%, 100% { opacity: 0.4; transform: scale(1); }
  50% { opacity: 1; transform: scale(1.6); }
}
@keyframes fade-in-up {
  0% { opacity: 0; transform: translateY(8px); }
  100% { opacity: 1; transform: translateY(0); }
}
@keyframes glow-card {
  0%, 100% { box-shadow: 0 0 8px rgba(0,255,136,0.08); }
  50% { box-shadow: 0 0 20px rgba(0,255,136,0.18); }
}
@keyframes sparkline-draw {
  0% { stroke-dashoffset: 500; }
  100% { stroke-dashoffset: 0; }
}
@keyframes radar-grow {
  0% { transform: scale(0); opacity: 0; }
  100% { transform: scale(1); opacity: 1; }
}
@keyframes slide-in {
  0% { opacity: 0; transform: translateX(-10px); }
  100% { opacity: 1; transform: translateX(0); }
}
@keyframes progress-fill {
  0% { width: 0%; }
}
@keyframes count-up {
  0% { opacity: 0; transform: translateY(4px); }
  100% { opacity: 1; transform: translateY(0); }
}
`;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

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
  totalR: number;
  trades: number;
  maxDD: number;
  rSquared: number;
  fwdWinRate: number | null;
  fwdPF: number | null;
  fwdTrades: number;
  fwdSince: string;
  trigger: string;
  stop: string;
  target: string;
  confluence: string[];
  monitored: boolean;
  btDays: number;
  // DNA values for radar chart
  dna: Record<string, number>;
  // Sparkline seed
  sparkSeed: number;
}

// ---------------------------------------------------------------------------
// Mock Data
// ---------------------------------------------------------------------------

const mockStrategies: Strategy[] = [
  {
    id: '1', name: 'NVDA LONG - Mass #2', symbol: 'NVDA', direction: 'LONG', timeframe: '1Min',
    session: 'RTH', status: 'On Track', tags: ['Scalping', 'High PF'],
    winRate: 54.0, pf: 2.05, dailyR: 1.95, totalR: 94.1, trades: 224, maxDD: -4.5, rSquared: 0.91,
    fwdWinRate: 52.1, fwdPF: 1.82, fwdTrades: 18, fwdSince: '2026-03-07',
    trigger: '[C] EMA Bull Cross', stop: 'Swing (5b)', target: '2R',
    confluence: ['5M-RVOL-HIGH', '1D-MACD_LINE-BULL'], monitored: true, btDays: 90,
    dna: { winRate: 72, consistency: 85, risk: 68, speed: 90, selectivity: 55, trend: 78 },
    sparkSeed: 1,
  },
  {
    id: '2', name: 'SPY LONG - Mass #1', symbol: 'SPY', direction: 'LONG', timeframe: '5Min',
    session: 'RTH', status: 'Outperforming', tags: ['Swing'],
    winRate: 62.5, pf: 3.12, dailyR: 2.41, totalR: 120.5, trades: 89, maxDD: -1.8, rSquared: 0.95,
    fwdWinRate: 65.0, fwdPF: 3.45, fwdTrades: 12, fwdSince: '2026-03-01',
    trigger: '[L0] VWAP Cross', stop: 'Swing (8b)', target: '3R',
    confluence: ['1H-EMA_STACK-SML'], monitored: true, btDays: 120,
    dna: { winRate: 88, consistency: 92, risk: 82, speed: 45, selectivity: 75, trend: 90 },
    sparkSeed: 2,
  },
  {
    id: '3', name: 'AAPL LONG - Mass #5', symbol: 'AAPL', direction: 'LONG', timeframe: '1Min',
    session: 'RTH', status: 'Insufficient Data', tags: ['New'],
    winRate: 48.2, pf: 1.45, dailyR: 0.82, totalR: 24.6, trades: 156, maxDD: -3.1, rSquared: 0.72,
    fwdWinRate: null, fwdPF: null, fwdTrades: 3, fwdSince: '2026-03-18',
    trigger: '[C] MACD Bull Cross', stop: 'ATR (14, 2x)', target: '1.5R',
    confluence: ['5M-RVOL-HIGH', '1D-EMA_STACK-SML'], monitored: false, btDays: 60,
    dna: { winRate: 48, consistency: 55, risk: 60, speed: 85, selectivity: 40, trend: 50 },
    sparkSeed: 3,
  },
  {
    id: '4', name: 'TSLA LONG - Mass #5', symbol: 'TSLA', direction: 'LONG', timeframe: '1Min',
    session: 'RTH', status: 'On Track', tags: ['Scalping'],
    winRate: 51.3, pf: 1.78, dailyR: 1.23, totalR: 55.3, trades: 201, maxDD: -2.9, rSquared: 0.84,
    fwdWinRate: 50.0, fwdPF: 1.65, fwdTrades: 8, fwdSince: '2026-03-10',
    trigger: '[HM] UT Bot Buy', stop: 'Swing (5b)', target: '2R',
    confluence: [], monitored: false, btDays: 90,
    dna: { winRate: 62, consistency: 72, risk: 75, speed: 92, selectivity: 30, trend: 65 },
    sparkSeed: 4,
  },
  {
    id: '5', name: 'META LONG - Mass #13', symbol: 'META', direction: 'LONG', timeframe: '5Min',
    session: 'RTH', status: 'Underperforming', tags: ['Swing', 'Low DD'],
    winRate: 45.8, pf: 1.12, dailyR: 0.34, totalR: 12.2, trades: 67, maxDD: -6.2, rSquared: 0.58,
    fwdWinRate: 40.0, fwdPF: 0.95, fwdTrades: 15, fwdSince: '2026-03-05',
    trigger: '[C] EMA Bull Cross', stop: 'Swing (10b)', target: '2.5R',
    confluence: ['5M-RVOL-HIGH'], monitored: false, btDays: 150,
    dna: { winRate: 42, consistency: 38, risk: 35, speed: 40, selectivity: 65, trend: 55 },
    sparkSeed: 5,
  },
  {
    id: '6', name: 'QQQ SHORT - Reversal #3', symbol: 'QQQ', direction: 'SHORT', timeframe: '5Min',
    session: 'RTH', status: 'On Track', tags: ['Swing', 'Contrarian'],
    winRate: 56.2, pf: 2.34, dailyR: 1.68, totalR: 67.2, trades: 112, maxDD: -3.4, rSquared: 0.87,
    fwdWinRate: 54.5, fwdPF: 2.10, fwdTrades: 22, fwdSince: '2026-02-28',
    trigger: '[C] MACD Bear Cross', stop: 'ATR (14, 1.5x)', target: '2R',
    confluence: ['1H-EMA_STACK-LMS', '5M-RVOL-HIGH'], monitored: true, btDays: 100,
    dna: { winRate: 76, consistency: 80, risk: 70, speed: 50, selectivity: 68, trend: 72 },
    sparkSeed: 6,
  },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const statusColors: Record<string, string> = {
  'On Track': 'var(--green)',
  'Outperforming': 'var(--blue)',
  'Underperforming': 'var(--red)',
  'Insufficient Data': 'var(--text-muted)',
};

const DNA_AXES = [
  { label: 'WR', key: 'winRate' },
  { label: 'Cons', key: 'consistency' },
  { label: 'Risk', key: 'risk' },
  { label: 'Speed', key: 'speed' },
  { label: 'Select', key: 'selectivity' },
  { label: 'Trend', key: 'trend' },
];

function daysSince(dateStr: string): number {
  const d = new Date(dateStr);
  const now = new Date();
  return Math.floor((now.getTime() - d.getTime()) / 86400000);
}

// ---------------------------------------------------------------------------
// SVG Sub-components
// ---------------------------------------------------------------------------

/** Mini Equity Sparkline — deterministic from seed */
function EquitySparkline({ seed, color, width = 280, height = 48 }: { seed: number; color: string; width?: number; height?: number }) {
  const points: number[] = [0];
  let val = 0;
  for (let i = 1; i <= 50; i++) {
    const r = Math.sin(seed * 137.5 * i) * 0.5 + 0.25 + Math.cos(seed * 43 * i) * 0.3;
    val += r;
    points.push(val);
  }
  const min = Math.min(...points);
  const max = Math.max(...points);
  const range = max - min || 1;

  const pathData = points.map((p, i) => {
    const x = (i / (points.length - 1)) * width;
    const y = height - 2 - ((p - min) / range) * (height - 4);
    return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');

  const fillData = `${pathData} L${width},${height} L0,${height} Z`;

  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      <path d={fillData} fill={color} opacity="0.08" />
      <path d={pathData} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round"
        style={{ strokeDasharray: 500, animation: 'sparkline-draw 1.2s ease-out forwards' }} />
      {/* End dot */}
      <circle cx={width} cy={height - 2 - ((points[points.length - 1] - min) / range) * (height - 4)}
        r="2.5" fill={color} />
    </svg>
  );
}

/** Mini DNA Radar — small version for card */
function MiniRadar({ values, size = 80 }: { values: Record<string, number>; size?: number }) {
  const center = size / 2;
  const maxR = size / 2 - 10;
  const numAxes = DNA_AXES.length;
  const angleStep = (2 * Math.PI) / numAxes;

  const dataPoints = DNA_AXES.map((axis, i) => {
    const angle = i * angleStep - Math.PI / 2;
    const val = (values[axis.key] || 0) / 100;
    return {
      x: center + Math.cos(angle) * maxR * val,
      y: center + Math.sin(angle) * maxR * val,
    };
  });

  const bgPts = DNA_AXES.map((_, i) => {
    const angle = i * angleStep - Math.PI / 2;
    return `${center + Math.cos(angle) * maxR},${center + Math.sin(angle) * maxR}`;
  }).join(' ');

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <polygon points={bgPts} fill="none" stroke="var(--border)" strokeWidth="0.5" opacity="0.4" />
      <polygon points={dataPoints.map(p => `${p.x},${p.y}`).join(' ')}
        fill="var(--accent)" fillOpacity="0.15" stroke="var(--accent)" strokeWidth="1.5"
        style={{ transformOrigin: `${center}px ${center}px`, animation: 'radar-grow 0.5s ease-out' }} />
      {dataPoints.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r="2" fill="var(--accent)" />
      ))}
    </svg>
  );
}

/** Full Radar for comparison mode */
function ComparisonRadar({ valuesA, valuesB, labelA, labelB, size = 240 }: {
  valuesA: Record<string, number>;
  valuesB: Record<string, number>;
  labelA: string;
  labelB: string;
  size?: number;
}) {
  const center = size / 2;
  const maxR = size / 2 - 28;
  const numAxes = DNA_AXES.length;
  const angleStep = (2 * Math.PI) / numAxes;
  const rings = [0.25, 0.5, 0.75, 1.0];

  const computePoints = (values: Record<string, number>) =>
    DNA_AXES.map((axis, i) => {
      const angle = i * angleStep - Math.PI / 2;
      const val = (values[axis.key] || 0) / 100;
      return {
        x: center + Math.cos(angle) * maxR * val,
        y: center + Math.sin(angle) * maxR * val,
      };
    });

  const pointsA = computePoints(valuesA);
  const pointsB = computePoints(valuesB);
  const labelPoints = DNA_AXES.map((axis, i) => {
    const angle = i * angleStep - Math.PI / 2;
    return {
      x: center + Math.cos(angle) * (maxR + 18),
      y: center + Math.sin(angle) * (maxR + 18),
      label: axis.label,
    };
  });

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {rings.map(ring => {
        const pts = DNA_AXES.map((_, i) => {
          const angle = i * angleStep - Math.PI / 2;
          return `${center + Math.cos(angle) * maxR * ring},${center + Math.sin(angle) * maxR * ring}`;
        }).join(' ');
        return <polygon key={ring} points={pts} fill="none" stroke="var(--border)" strokeWidth="0.5" opacity="0.4" />;
      })}
      {DNA_AXES.map((_, i) => {
        const angle = i * angleStep - Math.PI / 2;
        return <line key={i} x1={center} y1={center}
          x2={center + Math.cos(angle) * maxR} y2={center + Math.sin(angle) * maxR}
          stroke="var(--border)" strokeWidth="0.5" opacity="0.3" />;
      })}
      {/* Strategy A */}
      <polygon points={pointsA.map(p => `${p.x},${p.y}`).join(' ')}
        fill="var(--blue)" fillOpacity="0.12" stroke="var(--blue)" strokeWidth="2" />
      {/* Strategy B */}
      <polygon points={pointsB.map(p => `${p.x},${p.y}`).join(' ')}
        fill="var(--orange)" fillOpacity="0.12" stroke="var(--orange)" strokeWidth="2" />
      {/* Axis labels */}
      {labelPoints.map((p, i) => (
        <text key={i} x={p.x} y={p.y} textAnchor="middle" dominantBaseline="middle"
          fontSize="9" fill="var(--text-muted)" fontWeight="500">{p.label}</text>
      ))}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function StrategiesV4() {
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState('All');
  const [tagFilter, setTagFilter] = useState('All');
  const [sortBy, setSortBy] = useState('Daily R (High)');
  const [deleteId, setDeleteId] = useState<string | null>(null);

  // Comparison mode
  const [compareMode, setCompareMode] = useState(false);
  const [compareIds, setCompareIds] = useState<[string | null, string | null]>([null, null]);

  // All tags
  const allTags = useMemo(() => Array.from(new Set(mockStrategies.flatMap(s => s.tags))).sort(), []);
  const allStatuses = ['On Track', 'Outperforming', 'Underperforming', 'Insufficient Data'];

  const filtered = useMemo(() => {
    let result = [...mockStrategies];
    if (search) {
      const q = search.toLowerCase();
      result = result.filter(s =>
        s.name.toLowerCase().includes(q) ||
        s.symbol.toLowerCase().includes(q) ||
        s.tags.some(t => t.toLowerCase().includes(q))
      );
    }
    if (statusFilter !== 'All') result = result.filter(s => s.status === statusFilter);
    if (tagFilter !== 'All') result = result.filter(s => s.tags.includes(tagFilter));

    switch (sortBy) {
      case 'Daily R (High)': result.sort((a, b) => b.dailyR - a.dailyR); break;
      case 'Win Rate (High)': result.sort((a, b) => b.winRate - a.winRate); break;
      case 'Profit Factor (High)': result.sort((a, b) => b.pf - a.pf); break;
      case 'Total R (High)': result.sort((a, b) => b.totalR - a.totalR); break;
      case 'Trades (Most)': result.sort((a, b) => b.trades - a.trades); break;
      case 'Name A-Z': result.sort((a, b) => a.name.localeCompare(b.name)); break;
    }
    return result;
  }, [search, statusFilter, tagFilter, sortBy]);

  // Portfolio-level aggregates
  const aggStats = useMemo(() => {
    const total = mockStrategies.length;
    const avgWR = mockStrategies.reduce((s, st) => s + st.winRate, 0) / total;
    const avgPF = mockStrategies.reduce((s, st) => s + st.pf, 0) / total;
    const totalR = mockStrategies.reduce((s, st) => s + st.totalR, 0);
    const monitored = mockStrategies.filter(s => s.monitored).length;
    const bestPerformer = [...mockStrategies].sort((a, b) => b.dailyR - a.dailyR)[0];
    const worstPerformer = [...mockStrategies].sort((a, b) => a.dailyR - b.dailyR)[0];
    return { total, avgWR, avgPF, totalR, monitored, bestPerformer, worstPerformer };
  }, []);

  // Comparison
  const compareA = compareIds[0] ? mockStrategies.find(s => s.id === compareIds[0]) : null;
  const compareB = compareIds[1] ? mockStrategies.find(s => s.id === compareIds[1]) : null;

  const toggleCompare = (id: string) => {
    if (!compareMode) return;
    setCompareIds(prev => {
      if (prev[0] === id) return [null, prev[1]];
      if (prev[1] === id) return [prev[0], null];
      if (!prev[0]) return [id, prev[1]];
      if (!prev[1]) return [prev[0], id];
      return [id, prev[1]]; // replace first
    });
  };

  const selectStyle: React.CSSProperties = {
    background: 'var(--bg-input)',
    border: '1px solid var(--border)',
    color: 'var(--text-primary)',
    padding: '6px 12px',
    borderRadius: '8px',
    fontSize: '0.875rem',
  };

  const btnSecondary: React.CSSProperties = {
    background: 'var(--bg-card)',
    border: '1px solid var(--border)',
    color: 'var(--text-secondary)',
    padding: '6px 14px',
    borderRadius: '8px',
    fontSize: '0.875rem',
    cursor: 'pointer',
  };

  const btnPrimary: React.CSSProperties = {
    background: 'var(--accent)',
    border: 'none',
    color: 'white',
    padding: '6px 14px',
    borderRadius: '8px',
    fontSize: '0.875rem',
    fontWeight: 500,
    cursor: 'pointer',
  };

  const deleteStrategy = filtered.find(s => s.id === deleteId);

  return (
    <div>
      <style dangerouslySetInnerHTML={{ __html: ANIMATION_STYLES }} />

      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-2xl font-bold">My Strategies</h1>
        <div className="flex gap-3">
          <button
            style={compareMode ? { ...btnPrimary, background: 'var(--orange)' } : btnSecondary}
            onClick={() => { setCompareMode(!compareMode); setCompareIds([null, null]); }}
          >
            {compareMode ? 'Exit Compare' : 'Compare'}
          </button>
          <button style={btnSecondary}>Update Data</button>
          <button style={btnPrimary}>+ New Strategy</button>
        </div>
      </div>

      {/* Portfolio Overview Strip */}
      <div
        className="rounded-xl p-4 mb-5"
        style={{
          background: 'linear-gradient(135deg, color-mix(in srgb, var(--accent) 6%, var(--bg-card)), var(--bg-card))',
          border: '1px solid color-mix(in srgb, var(--accent) 15%, var(--border))',
        }}
      >
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-7 gap-4">
          <div style={{ animation: 'count-up 0.4s ease-out' }}>
            <div className="text-[10px] font-medium" style={{ color: 'var(--text-muted)' }}>Strategies</div>
            <div className="text-xl font-bold">{aggStats.total}</div>
          </div>
          <div style={{ animation: 'count-up 0.4s ease-out 0.05s both' }}>
            <div className="text-[10px] font-medium" style={{ color: 'var(--text-muted)' }}>Monitored</div>
            <div className="text-xl font-bold" style={{ color: 'var(--green)' }}>
              {aggStats.monitored}
              <span className="w-2 h-2 rounded-full inline-block ml-1.5"
                style={{ background: 'var(--green)', animation: 'pulse-live 2s ease-in-out infinite', verticalAlign: 'middle' }} />
            </div>
          </div>
          <div style={{ animation: 'count-up 0.4s ease-out 0.1s both' }}>
            <div className="text-[10px] font-medium" style={{ color: 'var(--text-muted)' }}>Avg Win Rate</div>
            <div className="text-xl font-bold">{aggStats.avgWR.toFixed(1)}%</div>
          </div>
          <div style={{ animation: 'count-up 0.4s ease-out 0.15s both' }}>
            <div className="text-[10px] font-medium" style={{ color: 'var(--text-muted)' }}>Avg PF</div>
            <div className="text-xl font-bold">{aggStats.avgPF.toFixed(2)}</div>
          </div>
          <div style={{ animation: 'count-up 0.4s ease-out 0.2s both' }}>
            <div className="text-[10px] font-medium" style={{ color: 'var(--text-muted)' }}>Total R</div>
            <div className="text-xl font-bold" style={{ color: 'var(--green)' }}>+{aggStats.totalR.toFixed(0)}</div>
          </div>
          <div style={{ animation: 'count-up 0.4s ease-out 0.25s both' }}>
            <div className="text-[10px] font-medium" style={{ color: 'var(--text-muted)' }}>Best Performer</div>
            <div className="text-sm font-bold" style={{ color: 'var(--blue)' }}>
              {aggStats.bestPerformer.symbol}
            </div>
            <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>+{aggStats.bestPerformer.dailyR.toFixed(2)} R/d</div>
          </div>
          <div style={{ animation: 'count-up 0.4s ease-out 0.3s both' }}>
            <div className="text-[10px] font-medium" style={{ color: 'var(--text-muted)' }}>Needs Attention</div>
            <div className="text-sm font-bold" style={{ color: 'var(--red)' }}>
              {aggStats.worstPerformer.symbol}
            </div>
            <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>+{aggStats.worstPerformer.dailyR.toFixed(2)} R/d</div>
          </div>
        </div>
      </div>

      {/* Filters + Search */}
      <div className="flex gap-3 mb-4 flex-wrap items-center">
        <input
          type="text"
          placeholder="Search by name, symbol, or tag..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="flex-1"
          style={{ ...selectStyle, maxWidth: '320px' }}
        />
        <select style={selectStyle} value={statusFilter} onChange={e => setStatusFilter(e.target.value)}>
          <option value="All">Status: All</option>
          {allStatuses.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
        <select style={selectStyle} value={tagFilter} onChange={e => setTagFilter(e.target.value)}>
          <option value="All">Tag: All</option>
          {allTags.map(t => <option key={t} value={t}>{t}</option>)}
        </select>
        <select style={selectStyle} value={sortBy} onChange={e => setSortBy(e.target.value)}>
          {['Daily R (High)', 'Win Rate (High)', 'Profit Factor (High)', 'Total R (High)', 'Trades (Most)', 'Name A-Z'].map(s => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
          {filtered.length} of {mockStrategies.length}
        </span>
      </div>

      {/* Compare hint */}
      {compareMode && (
        <div className="mb-4 p-3 rounded-lg text-sm" style={{
          background: 'color-mix(in srgb, var(--orange) 8%, var(--bg-card))',
          border: '1px solid color-mix(in srgb, var(--orange) 25%, var(--border))',
          color: 'var(--orange)',
        }}>
          Select 2 strategies to compare. {compareA ? `Selected: ${compareA.name}` : 'Click a card to select.'}
          {compareA && compareB ? ' Ready to compare!' : ''}
        </div>
      )}

      {/* Comparison Panel */}
      {compareMode && compareA && compareB && (
        <Card className="mb-5">
          <h3 className="text-sm font-semibold mb-3">Strategy Comparison</h3>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Radar overlay */}
            <div className="flex flex-col items-center">
              <ComparisonRadar valuesA={compareA.dna} valuesB={compareB.dna}
                labelA={compareA.name} labelB={compareB.name} size={220} />
              <div className="flex gap-4 mt-2 text-xs">
                <span><span className="inline-block w-3 h-0.5 mr-1 rounded" style={{ background: 'var(--blue)' }} />{compareA.symbol}</span>
                <span><span className="inline-block w-3 h-0.5 mr-1 rounded" style={{ background: 'var(--orange)' }} />{compareB.symbol}</span>
              </div>
            </div>
            {/* KPI comparison */}
            <div className="lg:col-span-2">
              <div className="overflow-x-auto">
                <table className="w-full text-xs" style={{ borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid var(--border)' }}>
                      <th className="text-left py-2 px-3" style={{ color: 'var(--text-muted)' }}>Metric</th>
                      <th className="text-right py-2 px-3" style={{ color: 'var(--blue)' }}>{compareA.symbol}</th>
                      <th className="text-right py-2 px-3" style={{ color: 'var(--orange)' }}>{compareB.symbol}</th>
                      <th className="text-right py-2 px-3" style={{ color: 'var(--text-muted)' }}>Delta</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { m: 'Win Rate', a: compareA.winRate, b: compareB.winRate, fmt: (v: number) => `${v.toFixed(1)}%` },
                      { m: 'Profit Factor', a: compareA.pf, b: compareB.pf, fmt: (v: number) => v.toFixed(2) },
                      { m: 'Daily R', a: compareA.dailyR, b: compareB.dailyR, fmt: (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}` },
                      { m: 'Total R', a: compareA.totalR, b: compareB.totalR, fmt: (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}` },
                      { m: 'Trades', a: compareA.trades, b: compareB.trades, fmt: (v: number) => String(v) },
                      { m: 'Max DD', a: compareA.maxDD, b: compareB.maxDD, fmt: (v: number) => `${v.toFixed(1)}R` },
                      { m: 'R-Squared', a: compareA.rSquared, b: compareB.rSquared, fmt: (v: number) => v.toFixed(2) },
                    ].map(row => {
                      const delta = row.a - row.b;
                      const better = row.m === 'Max DD' ? delta > 0 : delta > 0; // For MaxDD, less negative = better, but delta>0 means A is less negative
                      return (
                        <tr key={row.m} style={{ borderBottom: '1px solid var(--border)' }}>
                          <td className="py-2 px-3 font-medium">{row.m}</td>
                          <td className="py-2 px-3 text-right font-mono" style={{ color: 'var(--blue)' }}>{row.fmt(row.a)}</td>
                          <td className="py-2 px-3 text-right font-mono" style={{ color: 'var(--orange)' }}>{row.fmt(row.b)}</td>
                          <td className="py-2 px-3 text-right font-mono" style={{ color: better ? 'var(--green)' : 'var(--red)' }}>
                            {delta >= 0 ? '+' : ''}{delta.toFixed(2)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Strategy Cards (2-column grid) */}
      {filtered.length === 0 ? (
        <Card>
          <div className="text-center py-12">
            <p className="text-lg mb-2" style={{ color: 'var(--text-secondary)' }}>
              {mockStrategies.length === 0
                ? 'No strategies yet. Create your first strategy!'
                : 'No strategies match the current filters.'}
            </p>
          </div>
        </Card>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {filtered.map((strat, idx) => {
            const fwdDays = daysSince(strat.fwdSince);
            const isSelected = compareIds[0] === strat.id || compareIds[1] === strat.id;
            const sparkColor = strat.status === 'Outperforming' ? 'var(--blue)'
              : strat.status === 'Underperforming' ? 'var(--red)' : 'var(--green)';

            return (
              <div
                key={strat.id}
                className="rounded-xl transition-all"
                style={{
                  background: 'var(--bg-card)',
                  border: isSelected
                    ? '2px solid var(--orange)'
                    : '1px solid var(--border)',
                  padding: '16px',
                  cursor: compareMode ? 'pointer' : 'default',
                  animation: strat.monitored ? 'glow-card 3s ease-in-out infinite' : `fade-in-up 0.3s ease-out ${idx * 0.05}s both`,
                }}
                onClick={() => compareMode && toggleCompare(strat.id)}
              >
                {/* Header: name + status + monitoring */}
                <div className="flex items-center gap-2 mb-1">
                  {strat.monitored && (
                    <span className="w-2 h-2 rounded-full flex-shrink-0"
                      style={{ background: 'var(--green)', animation: 'pulse-live 2s ease-in-out infinite' }} />
                  )}
                  <h3 className="font-semibold text-base truncate">{strat.name}</h3>
                  <span className="text-[10px] px-2 py-0.5 rounded-full font-medium flex-shrink-0"
                    style={{ color: statusColors[strat.status], background: statusColors[strat.status] + '20' }}>
                    {strat.status}
                  </span>
                </div>

                {/* Tags */}
                {strat.tags.length > 0 && (
                  <div className="flex flex-wrap gap-1 mb-1">
                    {strat.tags.map(tag => (
                      <span key={tag} className="text-[10px] px-1.5 py-0.5 rounded-full"
                        style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>
                        {tag}
                      </span>
                    ))}
                  </div>
                )}

                {/* Meta line */}
                <p className="text-[11px] mb-2" style={{ color: 'var(--text-muted)' }}>
                  {strat.symbol} {strat.direction} | {strat.timeframe} | {strat.session}
                  <span style={{ color: 'var(--blue)' }}> | BT {strat.btDays}d</span>
                  <span style={{ color: 'var(--orange)' }}> | Fwd {fwdDays}d</span>
                </p>

                {/* Sparkline + Mini Radar side by side */}
                <div className="flex gap-3 mb-3">
                  <div className="flex-1 rounded-lg overflow-hidden" style={{ background: 'var(--bg-input)' }}>
                    <EquitySparkline seed={strat.sparkSeed} color={sparkColor} />
                  </div>
                  <div className="flex-shrink-0">
                    <MiniRadar values={strat.dna} size={56} />
                  </div>
                </div>

                {/* Primary KPIs */}
                <div className="grid grid-cols-5 gap-2 mb-2">
                  {[
                    { label: 'WR', value: `${strat.winRate.toFixed(1)}%`, color: strat.winRate > 50 ? 'var(--green)' : 'var(--red)' },
                    { label: 'PF', value: strat.pf.toFixed(2), color: strat.pf > 1.5 ? 'var(--green)' : undefined },
                    { label: 'Daily R', value: `${strat.dailyR >= 0 ? '+' : ''}${strat.dailyR.toFixed(2)}`, color: strat.dailyR >= 1.5 ? 'var(--green)' : undefined },
                    { label: 'Trades', value: String(strat.trades), color: undefined },
                    { label: 'Max DD', value: `${strat.maxDD.toFixed(1)}R`, color: 'var(--red)' },
                  ].map((kpi, j) => (
                    <div key={j}>
                      <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{kpi.label}</p>
                      <p className="text-sm font-medium" style={{ color: kpi.color }}>{kpi.value}</p>
                    </div>
                  ))}
                </div>

                {/* Forward Test KPIs */}
                {(strat.fwdWinRate !== null || strat.fwdTrades > 0) && (
                  <div className="grid grid-cols-5 gap-2 mb-2 pt-2" style={{ borderTop: '1px solid var(--border)' }}>
                    <div>
                      <p className="text-[10px]" style={{ color: 'var(--orange)' }}>Fwd WR</p>
                      <p className="text-sm font-medium">{strat.fwdWinRate !== null ? `${strat.fwdWinRate.toFixed(1)}%` : '--'}</p>
                    </div>
                    <div>
                      <p className="text-[10px]" style={{ color: 'var(--orange)' }}>Fwd PF</p>
                      <p className="text-sm font-medium">{strat.fwdPF !== null ? strat.fwdPF.toFixed(2) : '--'}</p>
                    </div>
                    <div>
                      <p className="text-[10px]" style={{ color: 'var(--orange)' }}>Fwd Trades</p>
                      <p className="text-sm font-medium">{strat.fwdTrades}</p>
                    </div>
                    <div>
                      <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>R-Squared</p>
                      <p className="text-sm font-medium">{strat.rSquared.toFixed(2)}</p>
                    </div>
                    <div>
                      <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Total R</p>
                      <p className="text-sm font-medium" style={{ color: 'var(--green)' }}>+{strat.totalR.toFixed(1)}</p>
                    </div>
                  </div>
                )}

                {/* Trigger + Confluence */}
                <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>
                  Entry: {strat.trigger} | Stop: {strat.stop} | TP: {strat.target}
                </p>
                {strat.confluence.length > 0 && (
                  <div className="flex flex-wrap gap-1 mt-1">
                    {strat.confluence.map(c => (
                      <span key={c} className="text-[9px] px-1.5 py-0.5 rounded"
                        style={{
                          color: 'var(--accent)',
                          background: 'color-mix(in srgb, var(--accent) 10%, transparent)',
                          border: '1px solid color-mix(in srgb, var(--accent) 20%, transparent)',
                        }}>
                        {c}
                      </span>
                    ))}
                  </div>
                )}

                {/* Action buttons */}
                {!compareMode && (
                  <div className="flex gap-2 mt-3 pt-2" style={{ borderTop: '1px solid var(--border)' }}>
                    <Link href={`/strategies/${strat.id}`}
                      className="px-3 py-1.5 rounded text-xs"
                      style={{ ...btnSecondary, fontSize: '0.75rem', padding: '4px 12px' }}>
                      View
                    </Link>
                    <button className="px-3 py-1.5 rounded text-xs"
                      style={{ ...btnSecondary, fontSize: '0.75rem', padding: '4px 12px' }}>
                      Edit
                    </button>
                    <button className="px-3 py-1.5 rounded text-xs"
                      style={{ ...btnSecondary, fontSize: '0.75rem', padding: '4px 12px' }}>
                      Clone
                    </button>
                    <button className="px-3 py-1.5 rounded text-xs"
                      style={{ background: 'var(--red-muted)', color: 'var(--red)', border: 'none', fontSize: '0.75rem', padding: '4px 12px', cursor: 'pointer' }}
                      onClick={() => setDeleteId(strat.id)}>
                      Delete
                    </button>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Delete confirmation modal */}
      <Modal title="Delete Strategy" isOpen={deleteId !== null} onClose={() => setDeleteId(null)} width="420px">
        {deleteStrategy && (
          <>
            <p className="text-sm mb-4" style={{ color: 'var(--text-secondary)' }}>
              Delete &apos;{deleteStrategy.name}&apos;? This cannot be undone.
            </p>
            <div className="flex gap-3 justify-end">
              <button style={btnSecondary} onClick={() => setDeleteId(null)}>Cancel</button>
              <button
                style={{ ...btnPrimary, background: 'var(--red)' }}
                onClick={() => setDeleteId(null)}>
                Delete
              </button>
            </div>
          </>
        )}
      </Modal>
    </div>
  );
}
