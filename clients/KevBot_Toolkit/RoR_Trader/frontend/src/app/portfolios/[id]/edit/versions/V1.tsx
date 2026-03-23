'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import MetricCard from '@/components/MetricCard';
import ChartPlaceholder from '@/components/ChartPlaceholder';

const AVAILABLE_STRATEGIES = [
  'NVDA LONG - Mass #2',
  'SPY LONG - Mass #1',
  'AAPL LONG - Mass #5',
  'TSLA LONG - Mass #5',
  'META LONG - Mass #13',
  'AMZN LONG - Mass #7',
];

const REQUIREMENT_SETS = ['None', 'TTP 50k', 'FTMO 100k'];

interface PortfolioStrategy {
  name: string;
  symbol: string;
  riskPerTrade: number;
}

const RECOMMENDATIONS = [
  { name: 'META LONG - Mass #13', symbol: 'META', pnlImpact: '+$420', ddImpact: '-0.3%' },
  { name: 'AMZN LONG - Mass #7', symbol: 'AMZN', pnlImpact: '+$310', ddImpact: '+0.1%' },
  { name: 'GOOGL LONG - Mass #4', symbol: 'GOOGL', pnlImpact: '+$185', ddImpact: '-0.1%' },
];

// Pre-filled mock data for editing
const INITIAL_PORTFOLIO = {
  name: 'My Portfolio',
  startingBalance: '80000',
  riskScaling: 85,
  requirementSet: 'TTP 50k',
  strategies: [
    { name: 'NVDA LONG - Mass #2', symbol: 'NVDA', riskPerTrade: 120 },
    { name: 'SPY LONG - Mass #1', symbol: 'SPY', riskPerTrade: 200 },
    { name: 'AAPL LONG - Mass #5', symbol: 'AAPL', riskPerTrade: 90 },
    { name: 'TSLA LONG - Mass #5', symbol: 'TSLA', riskPerTrade: 80 },
  ] as PortfolioStrategy[],
};

function extractSymbol(name: string): string {
  return name.split(' ')[0];
}

export default function PortfolioEditV1() {
  const router = useRouter();
  const [portfolioName, setPortfolioName] = useState(INITIAL_PORTFOLIO.name);
  const [startingBalance, setStartingBalance] = useState(INITIAL_PORTFOLIO.startingBalance);
  const [riskScaling, setRiskScaling] = useState(INITIAL_PORTFOLIO.riskScaling);
  const [requirementSet, setRequirementSet] = useState(INITIAL_PORTFOLIO.requirementSet);
  const [selectedStrategy, setSelectedStrategy] = useState('');
  const [riskPerTrade, setRiskPerTrade] = useState('100');
  const [strategies, setStrategies] = useState<PortfolioStrategy[]>([...INITIAL_PORTFOLIO.strategies]);
  const [showRecommendations, setShowRecommendations] = useState(true);

  const handleAddStrategy = () => {
    if (!selectedStrategy) return;
    if (strategies.find(s => s.name === selectedStrategy)) return;
    setStrategies([...strategies, {
      name: selectedStrategy,
      symbol: extractSymbol(selectedStrategy),
      riskPerTrade: Number(riskPerTrade) || 100,
    }]);
    setSelectedStrategy('');
    setRiskPerTrade('100');
  };

  const handleRemoveStrategy = (name: string) => {
    setStrategies(strategies.filter(s => s.name !== name));
  };

  const handleStrategyRiskChange = (name: string, newRisk: string) => {
    setStrategies(strategies.map(s =>
      s.name === name ? { ...s, riskPerTrade: Number(newRisk) || 0 } : s
    ));
  };

  const handleAddRecommendation = (rec: typeof RECOMMENDATIONS[0]) => {
    if (strategies.find(s => s.name === rec.name)) return;
    setStrategies([...strategies, {
      name: rec.name,
      symbol: rec.symbol,
      riskPerTrade: 100,
    }]);
  };

  const handleResetChanges = () => {
    setPortfolioName(INITIAL_PORTFOLIO.name);
    setStartingBalance(INITIAL_PORTFOLIO.startingBalance);
    setRiskScaling(INITIAL_PORTFOLIO.riskScaling);
    setRequirementSet(INITIAL_PORTFOLIO.requirementSet);
    setStrategies([...INITIAL_PORTFOLIO.strategies]);
  };

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

  // Filter out already-added strategies from dropdown
  const availableForAdd = AVAILABLE_STRATEGIES.filter(
    s => !strategies.find(st => st.name === s)
  );

  return (
    <div>
      <PageHeader
        title="Edit Portfolio"
        subtitle={`Modifying: ${INITIAL_PORTFOLIO.name}`}
        backHref="/portfolios/1"
      />

      {/* Settings Section */}
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Portfolio Settings</h3>
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
              placeholder="$25,000"
              value={startingBalance}
              onChange={(e) => setStartingBalance(e.target.value)}
              style={inputStyle}
            />
          </div>
          <div>
            <label className="text-xs block mb-1.5" style={{ color: 'var(--text-muted)' }}>
              Risk Scaling: {riskScaling}%
            </label>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <input
                type="range"
                min="0"
                max="100"
                value={riskScaling}
                onChange={(e) => setRiskScaling(Number(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--accent)' }}
              />
            </div>
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

      {/* Live Metrics Section */}
      <div className="grid grid-cols-6 gap-3 mb-6">
        <MetricCard label="Trades" value="847" />
        <MetricCard label="Win Rate" value="55.2%" />
        <MetricCard label="Profit Factor" value="1.89" />
        <MetricCard label="Total P&L" value="+$4,230" delta="+$4,230" positive />
        <MetricCard label="Final Balance" value="$84,230" />
        <MetricCard label="Max Drawdown" value="-2.1%" delta="-2.1%" positive={false} />
      </div>

      {/* Two-column layout */}
      <div className="grid grid-cols-5 gap-6 mb-6">
        {/* Left column - Charts */}
        <div className="col-span-3">
          <Card className="mb-4">
            <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Combined Equity Curve</h3>
            <div className="rounded-lg p-4" style={{ background: 'var(--bg-input)', height: 350 }}>
              <svg width="100%" height="100%" viewBox="0 0 500 280" preserveAspectRatio="none">
                <defs>
                  <linearGradient id="eqGradEdit" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.25" />
                    <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
                  </linearGradient>
                </defs>
                {/* Grid lines */}
                <line x1="0" y1="70" x2="500" y2="70" stroke="var(--border)" strokeWidth="0.5" strokeDasharray="4,4" />
                <line x1="0" y1="140" x2="500" y2="140" stroke="var(--border)" strokeWidth="0.5" strokeDasharray="4,4" />
                <line x1="0" y1="210" x2="500" y2="210" stroke="var(--border)" strokeWidth="0.5" strokeDasharray="4,4" />

                {/* Strategy 1 dotted line (NVDA) */}
                <path
                  d="M0,250 L25,245 L50,238 L75,240 L100,230 L125,225 L150,218 L175,222 L200,210 L225,205 L250,198 L275,192 L300,185 L325,180 L350,175 L375,168 L400,160 L425,155 L450,148 L475,142 L500,135"
                  fill="none"
                  stroke="var(--green)"
                  strokeWidth="1.5"
                  strokeDasharray="4,3"
                  opacity="0.6"
                />
                {/* Strategy 2 dotted line (SPY) */}
                <path
                  d="M0,250 L25,248 L50,242 L75,245 L100,235 L125,230 L150,228 L175,225 L200,218 L225,215 L250,210 L275,205 L300,198 L325,195 L350,190 L375,185 L400,180 L425,175 L450,170 L475,165 L500,160"
                  fill="none"
                  stroke="var(--blue)"
                  strokeWidth="1.5"
                  strokeDasharray="4,3"
                  opacity="0.6"
                />
                {/* Strategy 3 dotted line (AAPL) */}
                <path
                  d="M0,250 L25,248 L50,245 L75,242 L100,238 L125,235 L150,232 L175,228 L200,225 L225,220 L250,218 L275,215 L300,210 L325,208 L350,205 L375,200 L400,195 L425,192 L450,188 L475,185 L500,180"
                  fill="none"
                  stroke="var(--orange)"
                  strokeWidth="1.5"
                  strokeDasharray="4,3"
                  opacity="0.6"
                />
                {/* Strategy 4 dotted line (TSLA) */}
                <path
                  d="M0,250 L25,247 L50,244 L75,248 L100,240 L125,235 L150,230 L175,232 L200,225 L225,220 L250,215 L275,212 L300,208 L325,205 L350,200 L375,195 L400,190 L425,188 L450,185 L475,180 L500,175"
                  fill="none"
                  stroke="var(--red)"
                  strokeWidth="1.5"
                  strokeDasharray="4,3"
                  opacity="0.5"
                />
                {/* Combined equity curve - filled area */}
                <path
                  d="M0,250 L25,240 L50,225 L75,230 L100,210 L125,198 L150,185 L175,190 L200,172 L225,160 L250,148 L275,138 L300,125 L325,115 L350,105 L375,95 L400,82 L425,72 L450,60 L475,50 L500,40 L500,280 L0,280 Z"
                  fill="url(#eqGradEdit)"
                  stroke="none"
                />
                {/* Combined equity curve - solid line */}
                <path
                  d="M0,250 L25,240 L50,225 L75,230 L100,210 L125,198 L150,185 L175,190 L200,172 L225,160 L250,148 L275,138 L300,125 L325,115 L350,105 L375,95 L400,82 L425,72 L450,60 L475,50 L500,40"
                  fill="none"
                  stroke="var(--accent)"
                  strokeWidth="2.5"
                />
                {/* Legend */}
                <line x1="10" y1="15" x2="30" y2="15" stroke="var(--accent)" strokeWidth="2.5" />
                <text x="35" y="19" fill="var(--text-muted)" fontSize="9">Combined</text>
                <line x1="100" y1="15" x2="120" y2="15" stroke="var(--green)" strokeWidth="1.5" strokeDasharray="4,3" opacity="0.6" />
                <text x="125" y="19" fill="var(--text-muted)" fontSize="9">NVDA</text>
                <line x1="170" y1="15" x2="190" y2="15" stroke="var(--blue)" strokeWidth="1.5" strokeDasharray="4,3" opacity="0.6" />
                <text x="195" y="19" fill="var(--text-muted)" fontSize="9">SPY</text>
                <line x1="230" y1="15" x2="250" y2="15" stroke="var(--orange)" strokeWidth="1.5" strokeDasharray="4,3" opacity="0.6" />
                <text x="255" y="19" fill="var(--text-muted)" fontSize="9">AAPL</text>
                <line x1="300" y1="15" x2="320" y2="15" stroke="var(--red)" strokeWidth="1.5" strokeDasharray="4,3" opacity="0.5" />
                <text x="325" y="19" fill="var(--text-muted)" fontSize="9">TSLA</text>
              </svg>
            </div>
          </Card>

          <Card>
            <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Drawdown Analysis</h3>
            <ChartPlaceholder label="Drawdown curve — peaks at -2.1%" height={200} />
          </Card>
        </div>

        {/* Right column - Strategy Management */}
        <div className="col-span-2" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {/* Add Strategy */}
          <Card>
            <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Add Strategy</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <div>
                <label className="text-xs block mb-1.5" style={{ color: 'var(--text-muted)' }}>Strategy</label>
                <select
                  value={selectedStrategy}
                  onChange={(e) => setSelectedStrategy(e.target.value)}
                  style={selectStyle}
                >
                  <option value="">Select a strategy...</option>
                  {availableForAdd.map(s => (
                    <option key={s} value={s}>{s}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-xs block mb-1.5" style={{ color: 'var(--text-muted)' }}>Risk Per Trade ($)</label>
                <input
                  type="number"
                  placeholder="$100"
                  value={riskPerTrade}
                  onChange={(e) => setRiskPerTrade(e.target.value)}
                  style={inputStyle}
                />
              </div>
              <button
                onClick={handleAddStrategy}
                className="w-full py-2 rounded-lg text-sm font-medium transition-opacity"
                style={{
                  background: 'var(--accent)',
                  color: 'white',
                  opacity: selectedStrategy ? 1 : 0.5,
                  cursor: selectedStrategy ? 'pointer' : 'not-allowed',
                }}
                disabled={!selectedStrategy}
              >
                Add Strategy
              </button>
            </div>
          </Card>

          {/* Current Strategies */}
          <Card>
            <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>
              Current Strategies
              <span className="ml-2 text-xs" style={{ color: 'var(--text-muted)' }}>({strategies.length})</span>
            </h3>
            {strategies.length === 0 ? (
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No strategies added yet.</p>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {strategies.map((s) => (
                  <div
                    key={s.name}
                    className="rounded-lg p-3"
                    style={{
                      background: 'var(--bg-input)',
                      border: '1px solid var(--border)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      gap: '0.75rem',
                    }}
                  >
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span className="text-sm font-medium" style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{s.name}</span>
                        <span
                          className="text-xs px-1.5 py-0.5 rounded"
                          style={{ background: 'var(--accent-muted)', color: 'var(--accent)', whiteSpace: 'nowrap' }}
                        >
                          {s.symbol}
                        </span>
                      </div>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexShrink: 0 }}>
                      <span className="text-xs" style={{ color: 'var(--text-muted)' }}>$</span>
                      <input
                        type="number"
                        value={s.riskPerTrade}
                        onChange={(e) => handleStrategyRiskChange(s.name, e.target.value)}
                        style={{
                          ...inputStyle,
                          width: '70px',
                          padding: '0.25rem 0.5rem',
                          textAlign: 'right' as const,
                        }}
                      />
                      <span className="text-xs" style={{ color: 'var(--text-muted)' }}>/trade</span>
                      <button
                        onClick={() => handleRemoveStrategy(s.name)}
                        className="w-6 h-6 rounded flex items-center justify-center text-xs transition-colors"
                        style={{ color: 'var(--red)', background: 'transparent', flexShrink: 0 }}
                        onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--red-muted)')}
                        onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                        title="Remove strategy"
                      >
                        x
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </Card>

          {/* Recommendations */}
          <Card>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
              <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Recommendations</h3>
              <button
                onClick={() => setShowRecommendations(!showRecommendations)}
                className="text-xs px-2 py-1 rounded"
                style={{ color: 'var(--text-muted)', background: 'transparent' }}
              >
                {showRecommendations ? 'Hide' : 'Show'}
              </button>
            </div>

            {showRecommendations && (
              <div>
                <button
                  className="w-full py-1.5 rounded-lg text-xs font-medium mb-3"
                  style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
                >
                  Analyze Portfolio
                </button>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {RECOMMENDATIONS.filter(r => !strategies.find(s => s.name === r.name)).map((rec) => (
                    <div
                      key={rec.name}
                      className="rounded-lg p-3"
                      style={{
                        background: 'var(--bg-input)',
                        border: '1px solid var(--border)',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.375rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                          <span className="text-sm font-medium">{rec.name}</span>
                        </div>
                        <button
                          onClick={() => handleAddRecommendation(rec)}
                          className="text-xs px-2.5 py-1 rounded font-medium"
                          style={{ background: 'var(--accent)', color: 'white' }}
                        >
                          Add
                        </button>
                      </div>
                      <div style={{ display: 'flex', gap: '1rem' }}>
                        <span className="text-xs" style={{ color: 'var(--green)' }}>{rec.pnlImpact} P&L</span>
                        <span className="text-xs" style={{ color: rec.ddImpact.startsWith('+') ? 'var(--red)' : 'var(--green)' }}>
                          {rec.ddImpact} DD
                        </span>
                      </div>
                    </div>
                  ))}

                  {RECOMMENDATIONS.filter(r => !strategies.find(s => s.name === r.name)).length === 0 && (
                    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>All recommendations have been added.</p>
                  )}
                </div>
              </div>
            )}
          </Card>
        </div>
      </div>

      {/* Bottom Action Bar */}
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
          onClick={() => router.push('/portfolios/1')}
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
          onClick={handleResetChanges}
          className="px-4 py-2 rounded-lg text-sm font-medium"
          style={{
            background: 'var(--bg-card)',
            border: '1px solid var(--border)',
            color: 'var(--text-secondary)',
          }}
        >
          Reset Changes
        </button>
        <button
          className="px-6 py-2 rounded-lg text-sm font-medium"
          style={{ background: 'var(--accent)', color: 'white' }}
        >
          Update Portfolio
        </button>
      </div>
    </div>
  );
}
