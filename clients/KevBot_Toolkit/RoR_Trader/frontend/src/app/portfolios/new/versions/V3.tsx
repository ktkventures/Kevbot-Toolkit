'use client';

import { useState, useMemo, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import MetricCard from '@/components/MetricCard';

/* ========================================================================= */
/* MOCK DATA                                                                  */
/* ========================================================================= */

const STRATEGY_COLORS = ['var(--green)', 'var(--blue)', 'var(--orange)', 'var(--red)', '#AB47BC', '#26C6DA'];

interface StrategyData {
  id: string;
  name: string;
  symbol: string;
  direction: string;
  displayName: string;
  trades: number;
  winRate: number;
  pf: number;
  totalR: number;
  maxDD: number;
  avgR: number;
  equityCurve: number[];
}

const ALL_STRATEGIES: StrategyData[] = [
  {
    id: 's1', name: 'Mass #2', symbol: 'NVDA', direction: 'LONG',
    displayName: 'NVDA LONG - Mass #2',
    trades: 224, winRate: 54.0, pf: 2.05, totalR: 94.1, maxDD: -4.5, avgR: 0.42,
    equityCurve: [0, 5, 8, 6, 15, 20, 28, 25, 35, 40, 45, 52, 58, 65, 70, 78, 85, 92],
  },
  {
    id: 's2', name: 'Mass #1', symbol: 'SPY', direction: 'LONG',
    displayName: 'SPY LONG - Mass #1',
    trades: 186, winRate: 52.1, pf: 1.78, totalR: 61.2, maxDD: -3.8, avgR: 0.33,
    equityCurve: [0, 3, 5, 4, 10, 15, 18, 20, 25, 30, 32, 38, 42, 48, 52, 56, 62, 68],
  },
  {
    id: 's3', name: 'Mass #5', symbol: 'AAPL', direction: 'LONG',
    displayName: 'AAPL LONG - Mass #5',
    trades: 142, winRate: 56.3, pf: 1.92, totalR: 48.5, maxDD: -2.9, avgR: 0.34,
    equityCurve: [0, 2, 4, 6, 8, 12, 15, 18, 22, 28, 30, 35, 38, 42, 48, 52, 58, 62],
  },
  {
    id: 's4', name: 'Mass #5', symbol: 'TSLA', direction: 'LONG',
    displayName: 'TSLA LONG - Mass #5',
    trades: 198, winRate: 48.5, pf: 1.65, totalR: 38.2, maxDD: -6.1, avgR: 0.19,
    equityCurve: [0, 4, 2, 8, 5, 12, 18, 14, 22, 25, 20, 28, 32, 35, 30, 38, 42, 48],
  },
  {
    id: 's5', name: 'Mass #13', symbol: 'META', direction: 'LONG',
    displayName: 'META LONG - Mass #13',
    trades: 168, winRate: 55.4, pf: 1.88, totalR: 52.8, maxDD: -3.4, avgR: 0.31,
    equityCurve: [0, 3, 7, 10, 14, 18, 22, 25, 30, 34, 38, 42, 45, 50, 55, 60, 65, 70],
  },
  {
    id: 's6', name: 'Mass #7', symbol: 'AMZN', direction: 'LONG',
    displayName: 'AMZN LONG - Mass #7',
    trades: 155, winRate: 53.5, pf: 1.82, totalR: 44.6, maxDD: -3.1, avgR: 0.29,
    equityCurve: [0, 2, 5, 8, 12, 16, 20, 23, 26, 30, 34, 38, 40, 44, 48, 52, 56, 60],
  },
];

const REQUIREMENT_SETS = ['None', 'TTP 50k', 'FTMO 100k', 'Apex 150k'];

interface PortfolioStrategy {
  id: string;
  name: string;
  symbol: string;
  direction: string;
  displayName: string;
  riskPerTrade: number;
  trades: number;
  winRate: number;
  pf: number;
  totalR: number;
  maxDD: number;
  avgR: number;
  equityCurve: number[];
}

/* ========================================================================= */
/* HELPERS                                                                    */
/* ========================================================================= */

function buildEquityPath(curve: number[], maxVal: number, svgWidth: number, svgHeight: number, padTop: number, padBot: number): string {
  if (curve.length === 0) return '';
  const usableHeight = svgHeight - padTop - padBot;
  const stepX = svgWidth / (curve.length - 1);
  return curve.map((val, i) => {
    const x = i * stepX;
    const y = padTop + usableHeight - (val / maxVal) * usableHeight;
    return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
}

function buildCombinedCurve(strategies: PortfolioStrategy[], riskScaling: number): number[] {
  if (strategies.length === 0) return [];
  const maxLen = Math.max(...strategies.map(s => s.equityCurve.length));
  const combined: number[] = [];
  for (let i = 0; i < maxLen; i++) {
    let total = 0;
    for (const s of strategies) {
      const idx = Math.min(i, s.equityCurve.length - 1);
      const scaledVal = s.equityCurve[idx] * (s.riskPerTrade / 100);
      total += scaledVal * (1 + riskScaling / 100 * 0.5);
    }
    combined.push(total);
  }
  return combined;
}

/* ========================================================================= */
/* STYLES                                                                     */
/* ========================================================================= */

const inputStyle: React.CSSProperties = {
  background: 'var(--bg-input)',
  border: '1px solid var(--border)',
  color: 'var(--text-primary)',
  borderRadius: '0.5rem',
  padding: '0.5rem 0.75rem',
  fontSize: '0.875rem',
  width: '100%',
  outline: 'none',
};

const selectStyle: React.CSSProperties = {
  ...inputStyle,
  cursor: 'pointer',
  appearance: 'none' as const,
  backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%236b7280' d='M3 5l3 3 3-3'/%3E%3C/svg%3E")`,
  backgroundRepeat: 'no-repeat',
  backgroundPosition: 'right 0.75rem center',
  paddingRight: '2rem',
};

/* ========================================================================= */
/* COMPONENT                                                                  */
/* ========================================================================= */

export default function PortfolioNewV3() {
  const router = useRouter();

  // --- Settings state ---
  const [portfolioName, setPortfolioName] = useState('');
  const [startingBalance, setStartingBalance] = useState(25000);
  const [riskScaling, setRiskScaling] = useState(0);
  const [requirementSet, setRequirementSet] = useState('None');

  // --- Strategy builder state ---
  const [searchQuery, setSearchQuery] = useState('');
  const [strategies, setStrategies] = useState<PortfolioStrategy[]>([
    { ...ALL_STRATEGIES[0], riskPerTrade: 100 },
    { ...ALL_STRATEGIES[1], riskPerTrade: 150 },
    { ...ALL_STRATEGIES[2], riskPerTrade: 75 },
  ]);

  // --- Computed values ---
  const availableForAdd = useMemo(
    () => ALL_STRATEGIES.filter(s =>
      !strategies.find(st => st.id === s.id) &&
      (searchQuery === '' || s.displayName.toLowerCase().includes(searchQuery.toLowerCase()))
    ),
    [strategies, searchQuery]
  );

  const combinedCurve = useMemo(
    () => buildCombinedCurve(strategies, riskScaling),
    [strategies, riskScaling]
  );

  const metrics = useMemo(() => {
    if (strategies.length === 0) {
      return { trades: 0, winRate: 0, pf: 0, totalPnl: 0, finalBalance: 0, maxDD: 0 };
    }
    const totalTrades = strategies.reduce((sum, s) => sum + s.trades, 0);
    const weightedWR = strategies.reduce((sum, s) => sum + s.winRate * s.trades, 0) / totalTrades;
    const weightedPF = strategies.reduce((sum, s) => sum + s.pf * s.riskPerTrade, 0)
      / strategies.reduce((sum, s) => sum + s.riskPerTrade, 0);
    const totalPnl = strategies.reduce((sum, s) => sum + s.totalR * s.riskPerTrade, 0);
    const scaledPnl = totalPnl * (1 + riskScaling / 100 * 0.5);
    const finalBalance = startingBalance + scaledPnl;
    const maxDD = Math.min(...strategies.map(s => s.maxDD));
    return { trades: totalTrades, winRate: weightedWR, pf: weightedPF, totalPnl: scaledPnl, finalBalance, maxDD };
  }, [strategies, startingBalance, riskScaling]);

  // --- Handlers ---
  const handleAddStrategy = useCallback((stratId: string) => {
    const stratData = ALL_STRATEGIES.find(s => s.id === stratId);
    if (!stratData || strategies.find(s => s.id === stratId)) return;
    setStrategies(prev => [...prev, { ...stratData, riskPerTrade: 100 }]);
    setSearchQuery('');
  }, [strategies]);

  const handleRemoveStrategy = useCallback((id: string) => {
    setStrategies(prev => prev.filter(s => s.id !== id));
  }, []);

  const handleStrategyRiskChange = useCallback((id: string, newRisk: string) => {
    setStrategies(prev => prev.map(s =>
      s.id === id ? { ...s, riskPerTrade: Number(newRisk) || 0 } : s
    ));
  }, []);

  // --- SVG chart calculations ---
  const svgW = 500;
  const svgH = 200;
  const padTop = 20;
  const padBot = 10;
  const maxEquityVal = Math.max(1, ...combinedCurve);

  const combinedPath = buildEquityPath(combinedCurve, maxEquityVal, svgW, svgH, padTop, padBot);
  const combinedFillPath = combinedPath ? `${combinedPath} L${svgW},${svgH} L0,${svgH} Z` : '';

  return (
    <div>
      <PageHeader title="New Portfolio" backHref="/portfolios" />

      {/* ============ Settings Row (compact 4-column) ============ */}
      <Card className="mb-6">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
          <div>
            <label className="text-xs block mb-1.5" style={{ color: 'var(--text-muted)' }}>Portfolio Name</label>
            <input
              type="text"
              placeholder="My Portfolio"
              value={portfolioName}
              onChange={(e) => setPortfolioName(e.target.value)}
              style={inputStyle}
            />
          </div>
          <div>
            <label className="text-xs block mb-1.5" style={{ color: 'var(--text-muted)' }}>Starting Balance ($)</label>
            <input
              type="number"
              min={1000}
              step={1000}
              placeholder="25000"
              value={startingBalance}
              onChange={(e) => setStartingBalance(Number(e.target.value) || 1000)}
              style={inputStyle}
            />
          </div>
          <div>
            <label className="text-xs block mb-1.5" style={{ color: 'var(--text-muted)' }}>
              Risk Scaling: {riskScaling}%
            </label>
            <input
              type="range"
              min="0"
              max="100"
              step="5"
              value={riskScaling}
              onChange={(e) => setRiskScaling(Number(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--accent)' }}
            />
          </div>
          <div>
            <label className="text-xs block mb-1.5" style={{ color: 'var(--text-muted)' }}>Requirement Set</label>
            <select
              value={requirementSet}
              onChange={(e) => setRequirementSet(e.target.value)}
              style={selectStyle}
            >
              {REQUIREMENT_SETS.map(r => (
                <option key={r} value={r}>{r}</option>
              ))}
            </select>
          </div>
        </div>
      </Card>

      {/* ============ Inline KPI Strip ============ */}
      {strategies.length > 0 && (
        <div className="grid grid-cols-6 gap-3 mb-6">
          <MetricCard label="Trades" value={metrics.trades.toLocaleString()} />
          <MetricCard label="Win Rate" value={`${metrics.winRate.toFixed(1)}%`} />
          <MetricCard label="Profit Factor" value={metrics.pf.toFixed(2)} />
          <MetricCard
            label="Total P&L"
            value={`${metrics.totalPnl >= 0 ? '+' : ''}$${Math.abs(metrics.totalPnl).toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
            positive={metrics.totalPnl >= 0}
          />
          <MetricCard
            label="Final Balance"
            value={`$${metrics.finalBalance.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
          />
          <MetricCard
            label="Max Drawdown"
            value={`${metrics.maxDD.toFixed(1)}%`}
          />
        </div>
      )}

      {/* ============ Strategy List with Inline Add ============ */}
      <Card className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
            Strategies
            <span className="ml-2 text-xs" style={{ color: 'var(--text-muted)' }}>({strategies.length})</span>
          </h3>
        </div>

        {/* Inline Search + Add */}
        <div className="flex gap-2 mb-4">
          <div style={{ position: 'relative', flex: 1 }}>
            <input
              type="text"
              placeholder="Search strategies to add..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={inputStyle}
            />
            {/* Dropdown results */}
            {searchQuery && availableForAdd.length > 0 && (
              <div
                style={{
                  position: 'absolute',
                  top: '100%',
                  left: 0,
                  right: 0,
                  zIndex: 20,
                  background: 'var(--bg-card)',
                  border: '1px solid var(--border)',
                  borderRadius: '0.5rem',
                  marginTop: '0.25rem',
                  maxHeight: 200,
                  overflowY: 'auto',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
                }}
              >
                {availableForAdd.map(s => (
                  <button
                    key={s.id}
                    onClick={() => handleAddStrategy(s.id)}
                    className="w-full text-left px-3 py-2 text-sm transition-colors"
                    style={{ color: 'var(--text-primary)' }}
                    onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-input)')}
                    onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                  >
                    <span className="font-medium">{s.displayName}</span>
                    <span className="ml-2 text-xs" style={{ color: 'var(--text-muted)' }}>
                      {s.trades} trades | {s.winRate}% WR | PF {s.pf}
                    </span>
                  </button>
                ))}
              </div>
            )}
            {searchQuery && availableForAdd.length === 0 && (
              <div
                style={{
                  position: 'absolute',
                  top: '100%',
                  left: 0,
                  right: 0,
                  zIndex: 20,
                  background: 'var(--bg-card)',
                  border: '1px solid var(--border)',
                  borderRadius: '0.5rem',
                  marginTop: '0.25rem',
                  padding: '0.75rem 1rem',
                }}
              >
                <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No matching strategies found</p>
              </div>
            )}
          </div>
        </div>

        {/* Strategy rows: name, risk input, remove */}
        {strategies.length === 0 ? (
          <p className="text-sm text-center py-6" style={{ color: 'var(--text-muted)' }}>
            No strategies added yet. Search above to add strategies.
          </p>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {strategies.map((s, idx) => (
              <div
                key={s.id}
                className="flex items-center gap-3 rounded-lg p-3"
                style={{
                  background: 'var(--bg-input)',
                  border: '1px solid var(--border)',
                }}
              >
                {/* Color dot */}
                <div
                  style={{
                    width: 8, height: 8, borderRadius: '50%',
                    background: STRATEGY_COLORS[idx % STRATEGY_COLORS.length],
                    flexShrink: 0,
                  }}
                />

                {/* Name + symbol */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <span className="text-sm font-medium" style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', display: 'block' }}>
                    {s.displayName}
                  </span>
                </div>

                {/* Risk input */}
                <div className="flex items-center gap-1.5" style={{ flexShrink: 0 }}>
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>$</span>
                  <input
                    type="number"
                    min={1}
                    step={10}
                    value={s.riskPerTrade}
                    onChange={(e) => handleStrategyRiskChange(s.id, e.target.value)}
                    style={{
                      ...inputStyle,
                      width: '70px',
                      padding: '0.25rem 0.5rem',
                      textAlign: 'right' as const,
                    }}
                  />
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>/trade</span>
                </div>

                {/* Remove */}
                <button
                  onClick={() => handleRemoveStrategy(s.id)}
                  className="w-6 h-6 rounded flex items-center justify-center text-xs transition-colors"
                  style={{ color: 'var(--red)', background: 'transparent', flexShrink: 0 }}
                  onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--red-muted)')}
                  onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                  title="Remove strategy"
                >
                  x
                </button>
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* ============ Equity Preview ============ */}
      {strategies.length > 0 && (
        <Card className="mb-6">
          <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Combined Equity Preview</h3>
          <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)', height: 220 }}>
            <svg width="100%" height="100%" viewBox={`0 0 ${svgW} ${svgH}`} preserveAspectRatio="none">
              <defs>
                <linearGradient id="eqGradNewV3" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.25" />
                  <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
                </linearGradient>
              </defs>
              {/* Grid lines */}
              <line x1="0" y1="50" x2={svgW} y2="50" stroke="var(--border)" strokeWidth="0.5" strokeDasharray="4,4" />
              <line x1="0" y1="100" x2={svgW} y2="100" stroke="var(--border)" strokeWidth="0.5" strokeDasharray="4,4" />
              <line x1="0" y1="150" x2={svgW} y2="150" stroke="var(--border)" strokeWidth="0.5" strokeDasharray="4,4" />
              {/* Fill */}
              {combinedFillPath && (
                <path d={combinedFillPath} fill="url(#eqGradNewV3)" stroke="none" />
              )}
              {/* Line */}
              {combinedPath && (
                <path d={combinedPath} fill="none" stroke="var(--accent)" strokeWidth="2" />
              )}
            </svg>
          </div>
        </Card>
      )}

      {/* ============ Bottom Action Bar ============ */}
      <div
        className="rounded-xl border p-4"
        style={{
          background: 'var(--bg-card)',
          borderColor: 'var(--border)',
          display: 'flex',
          justifyContent: 'flex-end',
          gap: '0.75rem',
        }}
      >
        <button
          onClick={() => router.push('/portfolios')}
          className="px-4 py-2 rounded-lg text-sm font-medium"
          style={{
            background: 'var(--bg-card)',
            border: '1px solid var(--border)',
            color: 'var(--text-secondary)',
          }}
        >
          Cancel
        </button>
        <button
          className="px-6 py-2 rounded-lg text-sm font-medium transition-opacity"
          style={{
            background: 'var(--accent)',
            color: 'white',
            opacity: strategies.length > 0 && portfolioName.trim() ? 1 : 0.5,
            cursor: strategies.length > 0 && portfolioName.trim() ? 'pointer' : 'not-allowed',
          }}
          disabled={strategies.length === 0 || !portfolioName.trim()}
        >
          Save Portfolio
        </button>
      </div>
    </div>
  );
}
