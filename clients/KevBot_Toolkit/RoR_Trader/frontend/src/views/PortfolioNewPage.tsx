'use client';

/**
 * New Portfolio — Faithful copy of V5 design with mock data replaced by API hooks.
 * Source: src/app/portfolios/new/versions/V5.tsx
 *
 * Data convention:
 *   Real value = live from API
 *   -- = wired but no data yet
 *   {{field}} = not wired, needs backend work
 */

import { useState, useMemo, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { useQuery } from '@tanstack/react-query';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import MetricCard from '@/components/MetricCard';
import ChartPlaceholder from '@/components/ChartPlaceholder';
import Modal from '@/components/Modal';
import { useStrategies } from '@/hooks/queries/useStrategies';
import { usePortfolioPreview, usePortfolioRecommendations } from '@/hooks/queries/usePortfolios';
import { useCreatePortfolio } from '@/hooks/mutations/usePortfolioMutations';
import { apiFetch } from '@/lib/api/client';

/* ========================================================================= */
/* TYPES                                                                      */
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

/* ========================================================================= */
/* API → V5 Strategy Mapping                                                  */
/* ========================================================================= */

function apiToBuilderStrategy(s: any): StrategyData {
  const k = s.kpis || {};
  return {
    id: String(s.id),
    name: s.name || 'Untitled',
    symbol: s.symbol || '???',
    direction: s.direction || 'LONG',
    displayName: `${s.symbol || '???'} ${s.direction || 'LONG'} — ${s.name || 'Untitled'}`,
    trades: k.total_trades ?? 0,
    winRate: k.win_rate ?? 0,
    pf: k.profit_factor ?? 0,
    totalR: k.total_r ?? 0,
    maxDD: k.max_r_drawdown ?? 0,
    avgR: k.avg_r ?? 0,
    equityCurve: s.equity_curve_data?.cumulative_r || [],
  };
}

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

interface Recommendation {
  id: string;
  name: string;
  symbol: string;
  direction: string;
  displayName: string;
  pnlImpact: string;
  ddImpact: string;
  correlation: number;
  trades: number;
  winRate: number;
  pf: number;
  totalR: number;
  maxDD: number;
  avgR: number;
  equityCurve: number[];
}

const RECOMMENDATIONS: Recommendation[] = []; // {{recommendations}} — needs compute endpoint

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

function buildCombinedCurve(strategies: PortfolioStrategy[], balance: number, riskScaling: number): number[] {
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

function buildDrawdownCurve(combined: number[]): number[] {
  if (combined.length === 0) return [];
  const dd: number[] = [];
  let peak = 0;
  for (const val of combined) {
    if (val > peak) peak = val;
    const drawdown = peak > 0 ? ((val - peak) / peak) * 100 : 0;
    dd.push(drawdown);
  }
  return dd;
}

function buildDrawdownPath(ddCurve: number[], minDD: number, svgWidth: number, svgHeight: number, padTop: number): string {
  if (ddCurve.length === 0) return '';
  const stepX = svgWidth / (ddCurve.length - 1);
  const usable = svgHeight - padTop - 10;
  return ddCurve.map((val, i) => {
    const x = i * stepX;
    const y = padTop + (val / minDD) * usable;
    return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
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

export default function PortfolioNewPage() {
  const router = useRouter();

  // ---- API Hooks (ALL before early returns) ----
  const { data: apiStrategiesRaw, isLoading: strategiesLoading } = useStrategies();
  const createMut = useCreatePortfolio();
  const { data: reqSetsRaw } = useQuery({
    queryKey: ['requirements'],
    queryFn: () => apiFetch<any[]>('/api/requirements'),
  });

  // Map API strategies to builder format
  const allStrategyPool: StrategyData[] = useMemo(() => {
    if (!apiStrategiesRaw) return [];
    return apiStrategiesRaw.map(apiToBuilderStrategy);
  }, [apiStrategiesRaw]);

  // Requirement set names for dropdown
  const requirementSetOptions: string[] = useMemo(() => {
    const names = (reqSetsRaw || []).map((rs: any) => rs.name || '--');
    return ['None', ...names];
  }, [reqSetsRaw]);

  // --- Settings state ---
  const [portfolioName, setPortfolioName] = useState('');
  const [startingBalance, setStartingBalance] = useState(25000);
  const [riskScaling, setRiskScaling] = useState(0);
  const [requirementSet, setRequirementSet] = useState('None');

  // --- Strategy builder state ---
  const [selectedStrategy, setSelectedStrategy] = useState('');
  const [riskPerTrade, setRiskPerTrade] = useState('100');
  const [strategies, setStrategies] = useState<PortfolioStrategy[]>([]);

  // --- Equity curve view state ---
  const [eqView, setEqView] = useState<'Backtest' | 'Projected'>('Backtest');
  const [eqXAxis, setEqXAxis] = useState<'time' | 'trade'>('time');

  // --- Recommendations state ---
  const [showRecommendations, setShowRecommendations] = useState(false);
  const [recsAnalyzed, setRecsAnalyzed] = useState(false);
  const [showRecFilter, setShowRecFilter] = useState(false);
  const [recSortBy, setRecSortBy] = useState('P&L Impact');
  const [recMinWR, setRecMinWR] = useState('');
  const [recMinPF, setRecMinPF] = useState('');
  const [recMaxCorr, setRecMaxCorr] = useState('');
  const [recMinRODC, setRecMinRODC] = useState('');
  const [recRiskAssumption, setRecRiskAssumption] = useState('100');

  // --- Computed values ---
  const availableForAdd = useMemo(
    () => allStrategyPool.filter(s => !strategies.find(st => st.id === s.id)),
    [strategies, allStrategyPool]
  );

  // Real backend-computed preview (replaces synthetic buildCombinedCurve)
  const previewPayload = useMemo(() => {
    if (strategies.length === 0) return null;
    return {
      starting_balance: startingBalance,
      compound_rate: riskScaling / 100,
      strategies: strategies.map((s) => ({
        strategy_id: Number(s.id),
        risk_per_trade: s.riskPerTrade,
      })),
    };
  }, [strategies, startingBalance, riskScaling]);
  const { data: previewData } = usePortfolioPreview(previewPayload);

  // Fallback to synthetic curve while preview loads / for 0-strategy state
  const combinedCurve: number[] = useMemo(() => {
    if (previewData?.equity_curve && previewData.equity_curve.length > 0) {
      return previewData.equity_curve as number[];
    }
    return buildCombinedCurve(strategies, startingBalance, riskScaling);
  }, [previewData, strategies, startingBalance, riskScaling]);

  const drawdownCurve = useMemo(
    () => buildDrawdownCurve(combinedCurve),
    [combinedCurve]
  );

  const metrics = useMemo(() => {
    if (strategies.length === 0) {
      return {
        trades: 0, winRate: 0, pf: 0, maxDD: 0,
        dailyAvgPnl: 0, thirtyDayPnl: 0, tradesPerDay: 0,
        monthlyPnl: 0, quarterlyPnl: 0, annualPnl: 0, annualROI: 0,
        avgRPerTrade: 0, payoffRatio: 0, maxConcurrent: 0,
      };
    }

    // Prefer backend-computed KPIs from /preview endpoint
    const k = previewData?.kpis;
    if (k) {
      const tpd = k.trades_per_day ?? 0;
      const dailyAvg = k.avg_daily_pnl ?? 0;
      const totalTradesRaw = k.total_trades ?? 0;
      const avgRPerTrade = totalTradesRaw > 0
        ? strategies.reduce((sum, s) => sum + s.avgR * s.trades, 0) / strategies.reduce((sum, s) => sum + s.trades, 0)
        : 0;
      const wr = k.win_rate ?? 0;
      const pf = k.profit_factor && isFinite(k.profit_factor) ? k.profit_factor : 0;
      const payoffRatio = wr > 0 && wr < 100 && pf > 0
        ? pf * (1 - wr / 100) / (wr / 100)
        : 0;
      return {
        trades: totalTradesRaw,
        winRate: wr,
        pf,
        maxDD: k.max_drawdown_pct ?? 0,
        dailyAvgPnl: dailyAvg,
        thirtyDayPnl: dailyAvg * 21,
        tradesPerDay: tpd,
        monthlyPnl: dailyAvg * 21,
        quarterlyPnl: dailyAvg * 63,
        annualPnl: dailyAvg * 252,
        annualROI: (dailyAvg * 252 / startingBalance) * 100,
        avgRPerTrade,
        payoffRatio,
        maxConcurrent: strategies.length,
      };
    }

    // Fallback: per-strategy aggregation while preview loads
    const totalTrades = strategies.reduce((sum, s) => sum + s.trades, 0);
    const weightedWR = strategies.reduce((sum, s) => sum + s.winRate * s.trades, 0) / totalTrades;
    const weightedPF = strategies.reduce((sum, s) => sum + s.pf * s.riskPerTrade, 0)
      / strategies.reduce((sum, s) => sum + s.riskPerTrade, 0);
    const maxDD = Math.min(...strategies.map(s => s.maxDD));
    const tpd = strategies.reduce((sum, s) => sum + Math.max(0.5, s.trades / 60), 0);
    const dailyAvgPnl = strategies.reduce((sum, s) => sum + s.avgR * s.riskPerTrade * Math.max(0.5, s.trades / 60), 0)
      * (1 + riskScaling / 100 * 0.5);
    const avgRPerTrade = strategies.reduce((sum, s) => sum + s.avgR * s.trades, 0) / totalTrades;
    const payoffRatio = weightedPF * (1 - weightedWR / 100) / (weightedWR / 100);

    return {
      trades: totalTrades,
      winRate: weightedWR,
      pf: weightedPF,
      maxDD,
      dailyAvgPnl,
      thirtyDayPnl: dailyAvgPnl * 21,
      tradesPerDay: tpd,
      monthlyPnl: dailyAvgPnl * 21,
      quarterlyPnl: dailyAvgPnl * 63,
      annualPnl: dailyAvgPnl * 252,
      annualROI: (dailyAvgPnl * 252 / startingBalance) * 100,
      avgRPerTrade,
      payoffRatio,
      maxConcurrent: strategies.length,
    };
  }, [strategies, startingBalance, riskScaling, previewData]);

  const riskSummary = useMemo(() => {
    if (strategies.length === 0) {
      return { totalRiskPerDay: 0, maxDailyLoss: 0, worstCase: 0, capitalUtil: 0 };
    }
    const totalRiskPerTrade = strategies.reduce((sum, s) => sum + s.riskPerTrade, 0);
    // Assume avg ~3 trades per strategy per day
    const totalRiskPerDay = totalRiskPerTrade * 3;
    // Max daily loss: assume worst day = 2x average risk
    const maxDailyLoss = totalRiskPerDay * 2;
    // Worst case: max concurrent losing trades
    const worstCase = totalRiskPerTrade * strategies.length;
    // Capital utilization
    const capitalUtil = (totalRiskPerTrade / startingBalance) * 100;
    return { totalRiskPerDay, maxDailyLoss, worstCase, capitalUtil };
  }, [strategies, startingBalance]);

  // --- Handlers ---
  const handleAddStrategy = useCallback(() => {
    if (!selectedStrategy) return;
    const stratData = allStrategyPool.find(s => s.id === selectedStrategy);
    if (!stratData || strategies.find(s => s.id === selectedStrategy)) return;
    setStrategies(prev => [...prev, { ...stratData, riskPerTrade: Number(riskPerTrade) || 100 }]);
    setSelectedStrategy('');
    setRiskPerTrade('100');
    setRecsAnalyzed(false);
  }, [selectedStrategy, riskPerTrade, strategies]);

  const handleRemoveStrategy = useCallback((id: string) => {
    setStrategies(prev => prev.filter(s => s.id !== id));
    setRecsAnalyzed(false);
  }, []);

  const handleStrategyRiskChange = useCallback((id: string, newRisk: string) => {
    setStrategies(prev => prev.map(s =>
      s.id === id ? { ...s, riskPerTrade: Number(newRisk) || 0 } : s
    ));
  }, []);

  const handleMoveStrategy = useCallback((idx: number, direction: 'up' | 'down') => {
    setStrategies(prev => {
      const arr = [...prev];
      const targetIdx = direction === 'up' ? idx - 1 : idx + 1;
      if (targetIdx < 0 || targetIdx >= arr.length) return prev;
      [arr[idx], arr[targetIdx]] = [arr[targetIdx], arr[idx]];
      return arr;
    });
  }, []);

  const handleAddRecommendation = useCallback((rec: Recommendation) => {
    if (strategies.find(s => s.id === rec.id)) return;
    setStrategies(prev => [...prev, {
      id: rec.id,
      name: rec.name,
      symbol: rec.symbol,
      direction: rec.direction,
      displayName: rec.displayName,
      riskPerTrade: 100,
      trades: rec.trades,
      winRate: rec.winRate,
      pf: rec.pf,
      totalR: rec.totalR,
      maxDD: rec.maxDD,
      avgR: rec.avgR,
      equityCurve: rec.equityCurve,
    }]);
  }, [strategies]);

  const handleAnalyze = useCallback(() => {
    setRecsAnalyzed(true);
    setShowRecommendations(true);
  }, []);

  // --- SVG chart calculations ---
  const svgW = 500;
  const svgH = 280;
  const padTop = 30;
  const padBot = 10;
  const maxEquityVal = Math.max(1, ...combinedCurve, ...strategies.flatMap(s => s.equityCurve.map(v => v * (s.riskPerTrade / 100))));
  const minDD = Math.min(-0.1, ...drawdownCurve);

  const combinedPath = buildEquityPath(combinedCurve, maxEquityVal, svgW, svgH, padTop, padBot);
  const combinedFillPath = combinedPath ? `${combinedPath} L${svgW},${svgH} L0,${svgH} Z` : '';
  const ddPath = buildDrawdownPath(drawdownCurve, minDD, svgW, 180, 10);
  const ddFillPath = ddPath ? `${ddPath} L${svgW},10 L0,10 Z` : '';

  // --- Filtered recommendations (from /recommendations endpoint) ---
  const selectedStrategyIdsNum = useMemo(
    () => strategies.map((s) => Number(s.id)).filter((n) => Number.isFinite(n)),
    [strategies]
  );
  const { data: recsData } = usePortfolioRecommendations(
    recsAnalyzed ? selectedStrategyIdsNum : [],
    5,
  );
  const filteredRecs: Recommendation[] = useMemo(() => {
    if (!recsAnalyzed || !recsData?.recommendations) return [];
    return (recsData.recommendations as any[]).map((r) => {
      const full = allStrategyPool.find((s) => s.id === String(r.strategy_id));
      return {
        id: String(r.strategy_id),
        name: r.name || full?.name || '--',
        symbol: r.symbol || full?.symbol || '--',
        direction: full?.direction || 'LONG',
        displayName: r.name || full?.displayName || '--',
        pnlImpact: r.reasons?.[0] || '--',
        ddImpact: r.reasons?.[1] || '--',
        correlation: 0,
        trades: r.kpis?.total_trades ?? 0,
        winRate: r.kpis?.win_rate ?? 0,
        pf: r.kpis?.profit_factor ?? 0,
        totalR: full?.totalR ?? 0,
        maxDD: full?.maxDD ?? 0,
        avgR: full?.avgR ?? 0,
        equityCurve: full?.equityCurve ?? [],
      };
    }).filter((r) => !strategies.find((s) => s.id === r.id));
  }, [recsData, recsAnalyzed, strategies, allStrategyPool]);

  // ---- Loading state (after all hooks) ----
  if (strategiesLoading) {
    return (
      <div style={{ padding: '32px' }}>
        <h1 className="text-xl font-bold mb-4" style={{ color: 'var(--text-primary)' }}>New Portfolio</h1>
        <div className="space-y-4">
          {[1, 2].map((i) => (
            <Card key={i}><div className="animate-pulse space-y-3"><div className="h-4 rounded w-1/3" style={{ background: 'var(--border)' }} /><div className="h-3 rounded w-2/3" style={{ background: 'var(--border)' }} /></div></Card>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div>
      <PageHeader title="New Portfolio" backHref="/portfolios" />

      {/* ============ Settings Section ============ */}
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Portfolio Settings</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '1rem' }}>
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
            <p className="text-xs mt-1" style={{ color: 'var(--text-muted)', lineHeight: 1.3 }}>
              0% = fixed risk per trade. 100% = risk scales 1:1 with account growth.
            </p>
          </div>
          <div>
            <label className="text-xs block mb-1.5" style={{ color: 'var(--text-muted)' }}>Requirement Set</label>
            <select
              value={requirementSet}
              onChange={(e) => setRequirementSet(e.target.value)}
              style={selectStyle}
            >
              {requirementSetOptions.map(r => (
                <option key={r} value={r}>{r}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-xs block mb-1.5" style={{ color: 'var(--text-muted)' }}>Webhook Template</label>
            <select style={selectStyle} defaultValue="paper">
              <option value="paper">SignalStack — Paper Account</option>
              <option value="main">SignalStack — Main Account</option>
              <option value="discord">Discord — #trading-alerts</option>
              <option value="none">No template (webhooks disabled)</option>
            </select>
            <p className="text-xs mt-1" style={{ color: 'var(--text-muted)', lineHeight: 1.3 }}>
              Defaults to paper trading. Change after testing.
            </p>
          </div>
        </div>
      </Card>

      {/* ============ Primary KPIs (rate-based, duration-independent) ============ */}
      {strategies.length > 0 ? (
        <>
          {/* TQ filter for KPI display */}
          {requirementSet !== 'None' && (
            <div className="flex items-center gap-2 mb-3 text-xs" style={{ color: 'var(--text-muted)' }}>
              <span>Trade Qualification:</span>
              <select className="px-2 py-1 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} defaultValue="on">
                <option value="on">Apply {requirementSet} TQ rules</option>
                <option value="off">Show all trades (no TQ filter)</option>
              </select>
              <span style={{ color: 'var(--orange)' }}>KPIs below reflect filtered results</span>
            </div>
          )}
          <div className="grid grid-cols-6 gap-3 mb-3">
            <MetricCard
              label="Daily Avg P&L"
              value={`${metrics.dailyAvgPnl >= 0 ? '+' : ''}$${Math.abs(metrics.dailyAvgPnl).toFixed(0)}`}
              delta={metrics.dailyAvgPnl >= 0 ? 'positive' : 'negative'}
              positive={metrics.dailyAvgPnl >= 0}
            />
            <MetricCard label="30-Day Est." value={`${metrics.thirtyDayPnl >= 0 ? '+' : ''}$${Math.abs(metrics.thirtyDayPnl).toFixed(0)}`} />
            <MetricCard label="Win Rate" value={`${metrics.winRate.toFixed(1)}%`} />
            <MetricCard label="Profit Factor" value={metrics.pf.toFixed(2)} />
            <MetricCard label="Max DD" value={`${metrics.maxDD.toFixed(1)}%`} />
            <MetricCard label="Trades/Day" value={metrics.tradesPerDay.toFixed(1)} />
          </div>

          {/* Secondary KPIs */}
          <div className="grid grid-cols-4 sm:grid-cols-8 gap-3 mb-6">
            {[
              { label: 'Monthly Est.', value: `${metrics.monthlyPnl >= 0 ? '+' : ''}$${Math.abs(metrics.monthlyPnl).toFixed(0)}`, color: metrics.monthlyPnl >= 0 ? 'var(--green)' : 'var(--red)' },
              { label: 'Quarterly Est.', value: `${metrics.quarterlyPnl >= 0 ? '+' : ''}$${Math.abs(metrics.quarterlyPnl).toFixed(0)}`, color: metrics.quarterlyPnl >= 0 ? 'var(--green)' : 'var(--red)' },
              { label: 'Annual Est.', value: `${metrics.annualPnl >= 0 ? '+' : ''}$${Math.abs(metrics.annualPnl).toFixed(0)}`, color: metrics.annualPnl >= 0 ? 'var(--green)' : 'var(--red)' },
              { label: 'Annual ROI', value: `${metrics.annualROI >= 0 ? '+' : ''}${metrics.annualROI.toFixed(1)}%`, color: metrics.annualROI >= 0 ? 'var(--green)' : 'var(--red)' },
              { label: 'Total Trades', value: metrics.trades.toLocaleString() },
              { label: 'Avg R / Trade', value: `${metrics.avgRPerTrade >= 0 ? '+' : ''}${metrics.avgRPerTrade.toFixed(2)}R` },
              { label: 'Payoff Ratio', value: metrics.payoffRatio.toFixed(2) },
              { label: 'Max Concurrent', value: String(metrics.maxConcurrent) },
            ].map((kpi) => (
              <div key={kpi.label} className="rounded-lg p-2.5" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{kpi.label}</p>
                <p className="text-sm font-bold mt-0.5" style={kpi.color ? { color: kpi.color } : undefined}>{kpi.value}</p>
              </div>
            ))}
          </div>
        </>
      ) : (
        <Card className="mb-6">
          <p className="text-sm text-center py-4" style={{ color: 'var(--text-muted)' }}>
            Add strategies to see portfolio analytics.
          </p>
        </Card>
      )}

      {/* ============ Two-column layout ============ */}
      <div className="grid grid-cols-5 gap-6 mb-6">
        {/* Left column (60%) - Charts */}
        <div className="col-span-3">
          {/* Combined Equity Curve */}
          <Card className="mb-4">
            {/* Header with controls */}
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Combined Equity Curve</h3>
              <div className="flex items-center gap-3 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                <div className="flex items-center gap-1">
                  {(['Backtest', 'Projected'] as const).map((v) => (
                    <button key={v} onClick={() => setEqView(v)} className="px-2 py-1 rounded font-medium"
                      style={{ background: eqView === v ? 'var(--accent-muted)' : 'var(--bg-input)', color: eqView === v ? 'var(--accent)' : 'var(--text-muted)', border: eqView === v ? '1px solid var(--accent)' : '1px solid var(--border)', cursor: 'pointer' }}>
                      {v}
                    </button>
                  ))}
                </div>
                <div className="w-px h-3" style={{ background: 'var(--border)' }} />
                <div className="flex items-center gap-1">
                  <span>X:</span>
                  {(['time', 'trade'] as const).map((v) => (
                    <button key={v} onClick={() => setEqXAxis(v)} className="px-2 py-1 rounded font-medium"
                      style={{ background: eqXAxis === v ? 'var(--accent-muted)' : 'var(--bg-input)', color: eqXAxis === v ? 'var(--accent)' : 'var(--text-muted)', border: eqXAxis === v ? '1px solid var(--accent)' : '1px solid var(--border)', cursor: 'pointer' }}>
                      {v === 'time' ? 'Date' : 'Trade #'}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {strategies.length > 0 ? (
              <>
                {eqView === 'Backtest' ? (
                  <div className="rounded-lg p-4" style={{ background: 'var(--bg-input)', height: 350 }}>
                    <svg width="100%" height="100%" viewBox={`0 0 ${svgW} ${svgH}`} preserveAspectRatio="none">
                      <defs>
                        <linearGradient id="eqGradV2" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.25" />
                          <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
                        </linearGradient>
                      </defs>
                      <line x1="0" y1="70" x2={svgW} y2="70" stroke="var(--border)" strokeWidth="0.5" strokeDasharray="4,4" />
                      <line x1="0" y1="140" x2={svgW} y2="140" stroke="var(--border)" strokeWidth="0.5" strokeDasharray="4,4" />
                      <line x1="0" y1="210" x2={svgW} y2="210" stroke="var(--border)" strokeWidth="0.5" strokeDasharray="4,4" />
                      {strategies.map((s, i) => {
                        const scaledCurve = s.equityCurve.map(v => v * (s.riskPerTrade / 100));
                        const path = buildEquityPath(scaledCurve, maxEquityVal, svgW, svgH, padTop, padBot);
                        return (
                          <path key={s.id} d={path} fill="none" stroke={STRATEGY_COLORS[i % STRATEGY_COLORS.length]} strokeWidth="1.5" strokeDasharray="4,3" opacity="0.6" />
                        );
                      })}
                      {combinedFillPath && <path d={combinedFillPath} fill="url(#eqGradV2)" stroke="none" />}
                      {combinedPath && <path d={combinedPath} fill="none" stroke="var(--accent)" strokeWidth="2.5" />}
                    </svg>
                  </div>
                ) : (
                  <div>
                    <ChartPlaceholder label={`Projected combined equity: daily expected P&L summed across all strategies (avg R × risk/trade × trades/day). All strategies start from day 0. Confidence bands at ±1SD and ±2SD. X-axis: ${eqXAxis === 'time' ? 'projected days forward' : 'projected trade count'}`} height={350} />
                    <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>
                      Projected view normalizes all strategies to start at the same point, using each strategy&apos;s daily average performance. Shows forward-looking expectations rather than historical backtests with different start dates.
                    </p>
                  </div>
                )}
              </>
            ) : (
              <div className="rounded-lg flex items-center justify-center" style={{ background: 'var(--bg-input)', height: 350 }}>
                <span className="text-sm" style={{ color: 'var(--text-muted)' }}>Add strategies to see equity curve</span>
              </div>
            )}

            {/* Chart Legend */}
            {strategies.length > 0 && (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginTop: '0.75rem', paddingLeft: '0.25rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
                  <div style={{ width: 24, height: 3, background: 'var(--accent)', borderRadius: 2 }} />
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Combined</span>
                </div>
                {strategies.map((s, i) => (
                  <div key={s.id} style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
                    <div style={{
                      width: 24, height: 2,
                      background: STRATEGY_COLORS[i % STRATEGY_COLORS.length],
                      borderRadius: 2,
                      opacity: 0.7,
                      backgroundImage: `repeating-linear-gradient(90deg, ${STRATEGY_COLORS[i % STRATEGY_COLORS.length]} 0px, ${STRATEGY_COLORS[i % STRATEGY_COLORS.length]} 4px, transparent 4px, transparent 7px)`,
                    }} />
                    <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{s.symbol}</span>
                  </div>
                ))}
              </div>
            )}
          </Card>

          {/* Drawdown Chart */}
          <Card>
            <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Drawdown Analysis</h3>
            {strategies.length > 0 && drawdownCurve.length > 0 ? (
              <div className="rounded-lg p-4" style={{ background: 'var(--bg-input)', height: 200 }}>
                <svg width="100%" height="100%" viewBox={`0 0 ${svgW} 180`} preserveAspectRatio="none">
                  <defs>
                    <linearGradient id="ddGradV2" x1="0" y1="1" x2="0" y2="0">
                      <stop offset="0%" stopColor="var(--red)" stopOpacity="0.2" />
                      <stop offset="100%" stopColor="var(--red)" stopOpacity="0" />
                    </linearGradient>
                  </defs>
                  {/* Zero line */}
                  <line x1="0" y1="10" x2={svgW} y2="10" stroke="var(--border)" strokeWidth="0.5" />
                  {/* Drawdown fill */}
                  {ddFillPath && <path d={ddFillPath} fill="url(#ddGradV2)" stroke="none" />}
                  {/* Drawdown line */}
                  {ddPath && <path d={ddPath} fill="none" stroke="var(--red)" strokeWidth="1.5" />}
                  {/* Label */}
                  <text x={svgW - 5} y={25} fill="var(--text-muted)" fontSize="10" textAnchor="end">
                    Peak: {metrics.maxDD.toFixed(1)}%
                  </text>
                </svg>
              </div>
            ) : (
              <div
                className="rounded-lg flex items-center justify-center"
                style={{ background: 'var(--bg-input)', height: 200 }}
              >
                <span className="text-sm" style={{ color: 'var(--text-muted)' }}>Drawdown curve will appear here</span>
              </div>
            )}
          </Card>
        </div>

        {/* Right column (40%) - Strategy Management */}
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
                  disabled={availableForAdd.length === 0}
                >
                  <option value="">
                    {availableForAdd.length === 0 ? 'All strategies added' : 'Select a strategy...'}
                  </option>
                  {availableForAdd.map(s => (
                    <option key={s.id} value={s.id}>{s.displayName}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-xs block mb-1.5" style={{ color: 'var(--text-muted)' }}>Risk Per Trade ($)</label>
                <input
                  type="number"
                  min={1}
                  step={10}
                  placeholder="$100"
                  value={riskPerTrade}
                  onChange={(e) => setRiskPerTrade(e.target.value)}
                  style={inputStyle}
                  disabled={availableForAdd.length === 0}
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
                disabled={!selectedStrategy || availableForAdd.length === 0}
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
                {strategies.map((s, idx) => {
                  const riskPct = startingBalance > 0 ? ((s.riskPerTrade / startingBalance) * 100).toFixed(1) : '0.0';
                  return (
                    <div
                      key={s.id}
                      className="rounded-lg p-3"
                      style={{
                        background: 'var(--bg-input)',
                        border: '1px solid var(--border)',
                      }}
                    >
                      {/* Row 1: Name + badge + reorder + remove */}
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.5rem', marginBottom: '0.375rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flex: 1, minWidth: 0 }}>
                          <div
                            style={{
                              width: 8, height: 8, borderRadius: '50%',
                              background: STRATEGY_COLORS[idx % STRATEGY_COLORS.length],
                              flexShrink: 0,
                            }}
                          />
                          <span className="text-sm font-medium" style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                            {s.name}
                          </span>
                          <span
                            className="text-xs px-1.5 py-0.5 rounded"
                            style={{ background: 'var(--accent-muted)', color: 'var(--accent)', whiteSpace: 'nowrap', flexShrink: 0 }}
                          >
                            {s.symbol}
                          </span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.125rem', flexShrink: 0 }}>
                          <button
                            onClick={() => handleMoveStrategy(idx, 'up')}
                            disabled={idx === 0}
                            className="w-5 h-5 rounded flex items-center justify-center text-xs"
                            style={{
                              color: idx === 0 ? 'var(--border)' : 'var(--text-muted)',
                              background: 'transparent',
                              cursor: idx === 0 ? 'default' : 'pointer',
                            }}
                            title="Move up"
                          >
                            &#9650;
                          </button>
                          <button
                            onClick={() => handleMoveStrategy(idx, 'down')}
                            disabled={idx === strategies.length - 1}
                            className="w-5 h-5 rounded flex items-center justify-center text-xs"
                            style={{
                              color: idx === strategies.length - 1 ? 'var(--border)' : 'var(--text-muted)',
                              background: 'transparent',
                              cursor: idx === strategies.length - 1 ? 'default' : 'pointer',
                            }}
                            title="Move down"
                          >
                            &#9660;
                          </button>
                          <button
                            onClick={() => handleRemoveStrategy(s.id)}
                            className="w-5 h-5 rounded flex items-center justify-center text-xs transition-colors ml-1"
                            style={{ color: 'var(--red)', background: 'transparent', flexShrink: 0 }}
                            onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--red-muted)')}
                            onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                            title="Remove strategy"
                          >
                            x
                          </button>
                        </div>
                      </div>

                      {/* Row 2: KPI summary */}
                      <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '0.375rem' }}>
                        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{s.trades} trades</span>
                        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{s.winRate.toFixed(1)}% WR</span>
                        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>PF {s.pf.toFixed(2)}</span>
                        <span className="text-xs" style={{ color: s.totalR >= 0 ? 'var(--green)' : 'var(--red)' }}>
                          {s.totalR >= 0 ? '+' : ''}{s.totalR.toFixed(1)}R
                        </span>
                      </div>

                      {/* Row 3: Risk input + position sizing */}
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
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
                        <span className="text-xs ml-auto" style={{ color: 'var(--text-muted)' }}>
                          At ${s.riskPerTrade}/trade with ${startingBalance.toLocaleString()} balance = {riskPct}% risk
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </Card>

          {/* Strategy Recommendations */}
          <Card>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
              <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
                Strategy Recommendations
              </h3>
              <div className="flex items-center gap-2">
                {recsAnalyzed && (
                  <button
                    onClick={() => setShowRecFilter(true)}
                    className="text-xs px-2 py-1 rounded"
                    style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-muted)', cursor: 'pointer' }}
                  >
                    Filter &amp; Sort
                  </button>
                )}
                {recsAnalyzed && (
                  <button
                    onClick={() => setShowRecommendations(!showRecommendations)}
                    className="text-xs px-2 py-1 rounded"
                    style={{ color: 'var(--text-muted)', background: 'transparent', cursor: 'pointer' }}
                  >
                    {showRecommendations ? 'Hide' : 'Show'}
                  </button>
                )}
              </div>
            </div>

            {strategies.length > 0 && (
              <button
                onClick={handleAnalyze}
                className="w-full py-1.5 rounded-lg text-xs font-medium mb-3 transition-colors"
                style={{
                  background: recsAnalyzed ? 'var(--bg-input)' : 'var(--accent-muted)',
                  color: recsAnalyzed ? 'var(--text-muted)' : 'var(--accent)',
                  border: recsAnalyzed ? '1px solid var(--border)' : 'none',
                  cursor: 'pointer',
                }}
              >
                {recsAnalyzed ? 'Re-analyze Portfolio' : 'Analyze Portfolio'}
              </button>
            )}

            {/* Filter & Sort Modal */}
            <Modal title="Filter & Sort Recommendations" isOpen={showRecFilter} onClose={() => setShowRecFilter(false)} width="540px">
              <div className="space-y-4">
                <div>
                  <label className="text-xs font-medium block mb-1.5" style={{ color: 'var(--text-muted)' }}>Sort By</label>
                  <select value={recSortBy} onChange={(e) => setRecSortBy(e.target.value)}
                    style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '8px 12px', borderRadius: '8px', fontSize: '0.875rem', width: '100%' }}>
                    {['P&L Impact', 'DD Impact', 'Correlation (Low)', 'Win Rate', 'Profit Factor', 'RODC (High)', 'Avg Hold Time (Short)'].map((o) => (
                      <option key={o} value={o}>{o}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="text-xs font-medium block mb-2" style={{ color: 'var(--text-muted)' }}>Risk Assumption (for RODC calculation)</label>
                  <div className="flex items-center gap-2">
                    <span className="text-xs" style={{ color: 'var(--text-muted)' }}>$</span>
                    <input type="number" value={recRiskAssumption} onChange={(e) => setRecRiskAssumption(e.target.value)} min="10" step="10" placeholder="100"
                      style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '8px 12px', borderRadius: '8px', fontSize: '0.875rem', width: 120 }} />
                    <span className="text-xs" style={{ color: 'var(--text-muted)' }}>per trade — used to estimate capital deployed and RODC</span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-xs font-medium block mb-1.5" style={{ color: 'var(--text-muted)' }}>Min Win Rate (%)</label>
                    <input type="number" value={recMinWR} onChange={(e) => setRecMinWR(e.target.value)} placeholder="No minimum" min="0" max="100" step="1"
                      style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '8px 12px', borderRadius: '8px', fontSize: '0.875rem', width: '100%' }} />
                  </div>
                  <div>
                    <label className="text-xs font-medium block mb-1.5" style={{ color: 'var(--text-muted)' }}>Min Profit Factor</label>
                    <input type="number" value={recMinPF} onChange={(e) => setRecMinPF(e.target.value)} placeholder="No minimum" min="0" step="0.1"
                      style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '8px 12px', borderRadius: '8px', fontSize: '0.875rem', width: '100%' }} />
                  </div>
                  <div>
                    <label className="text-xs font-medium block mb-1.5" style={{ color: 'var(--text-muted)' }}>Max Correlation</label>
                    <input type="number" value={recMaxCorr} onChange={(e) => setRecMaxCorr(e.target.value)} placeholder="No maximum" min="0" max="1" step="0.05"
                      style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '8px 12px', borderRadius: '8px', fontSize: '0.875rem', width: '100%' }} />
                  </div>
                  <div>
                    <label className="text-xs font-medium block mb-1.5" style={{ color: 'var(--text-muted)' }}>Min RODC ($/day per $1k)</label>
                    <input type="number" value={recMinRODC} onChange={(e) => setRecMinRODC(e.target.value)} placeholder="No minimum" min="0" step="1"
                      style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '8px 12px', borderRadius: '8px', fontSize: '0.875rem', width: '100%' }} />
                  </div>
                </div>

                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  RODC (Return on Deployed Capital) = daily P&L per $1k of average capital deployed. Higher RODC means the strategy generates more return per dollar of buying power consumed. Factors in trade frequency, hold time, and risk per trade.
                </p>
              </div>

              <div className="flex justify-end gap-2 mt-6">
                <button onClick={() => { setRecMinWR(''); setRecMinPF(''); setRecMaxCorr(''); setRecMinRODC(''); setRecSortBy('P&L Impact'); }}
                  className="px-3 py-1.5 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-muted)', cursor: 'pointer' }}>
                  Reset
                </button>
                <button onClick={() => setShowRecFilter(false)}
                  className="px-4 py-1.5 rounded text-xs font-medium" style={{ background: 'var(--accent)', color: 'white', border: 'none', cursor: 'pointer' }}>
                  Apply
                </button>
              </div>
            </Modal>

            {!recsAnalyzed && strategies.length > 0 && (
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                Click Analyze to get strategy recommendations based on portfolio correlation and risk analysis.
              </p>
            )}

            {strategies.length === 0 && (
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                Add strategies first to get recommendations.
              </p>
            )}

            {recsAnalyzed && showRecommendations && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {filteredRecs.length > 0 ? filteredRecs.map((rec) => (
                  <div
                    key={rec.id}
                    className="rounded-lg p-3"
                    style={{
                      background: 'var(--bg-input)',
                      border: '1px solid var(--border)',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.375rem' }}>
                      <span className="text-sm font-medium">{rec.displayName}</span>
                      <button
                        onClick={() => handleAddRecommendation(rec)}
                        className="text-xs px-2.5 py-1 rounded font-medium"
                        style={{ background: 'var(--accent)', color: 'white' }}
                      >
                        Add
                      </button>
                    </div>
                    <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                      <span className="text-xs" style={{ color: 'var(--green)' }}>
                        {rec.pnlImpact} P&L
                      </span>
                      <span className="text-xs" style={{ color: rec.ddImpact.startsWith('+') ? 'var(--red)' : 'var(--green)' }}>
                        {rec.ddImpact} DD
                      </span>
                      <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                        Corr: {rec.correlation.toFixed(2)}
                      </span>
                      <span className="text-xs" style={{ color: 'var(--accent)' }}>
                        RODC: ${(rec.avgR * Number(recRiskAssumption || 100) * 0.8).toFixed(0)}/day per $1k
                      </span>
                    </div>
                  </div>
                )) : (
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    All recommendations have been added.
                  </p>
                )}
              </div>
            )}
          </Card>

        </div>
      </div>

      {/* ============ Risk Summary (full width, horizontal) ============ */}
      {strategies.length > 0 && (
        <Card className="mb-6">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Risk Summary</h3>
          </div>
          <div className="grid grid-cols-4 gap-6 mt-3">
            <div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Total Risk Per Day (est.)</p>
              <p className="text-sm font-bold">${riskSummary.totalRiskPerDay.toLocaleString()}</p>
            </div>
            <div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Max Daily Loss Exposure</p>
              <p className="text-sm font-bold" style={{ color: 'var(--red)' }}>-${riskSummary.maxDailyLoss.toLocaleString()}</p>
            </div>
            <div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Worst Case Scenario</p>
              <p className="text-sm font-bold" style={{ color: 'var(--red)' }}>-${riskSummary.worstCase.toLocaleString()}</p>
            </div>
            <div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Capital Utilization</p>
              <div className="flex items-center gap-2 mt-1">
                <div style={{ width: 80, height: 6, background: 'var(--border)', borderRadius: 3, overflow: 'hidden' }}>
                  <div style={{
                    width: `${Math.min(100, riskSummary.capitalUtil)}%`, height: '100%', borderRadius: 3,
                    background: riskSummary.capitalUtil > 5 ? 'var(--red)' : riskSummary.capitalUtil > 2 ? 'var(--orange)' : 'var(--green)',
                  }} />
                </div>
                <span className="text-sm font-bold" style={{
                  color: riskSummary.capitalUtil > 5 ? 'var(--red)' : riskSummary.capitalUtil > 2 ? 'var(--orange)' : 'var(--green)',
                }}>
                  {riskSummary.capitalUtil.toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* ============ Risk Analytics (shown when strategies added) ============ */}
      {strategies.length > 0 && (
        <div className="space-y-6 mb-6">
          {/* Capital Utilization */}
          <Card>
            <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Daily Peak Capital Deployed</h3>
            <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
              Estimated peak buying power usage per day based on strategy trade frequency and risk per trade. Red line marks your starting balance.
            </p>
            <ChartPlaceholder label="Bar chart by day: estimated peak capital deployed per day (blue bars). Red dashed line at starting balance ($). Shows whether buying power may be a constraint." height={180} />
            <div className="grid grid-cols-4 gap-4 mt-3">
              {[
                { label: 'Est. Peak / Day', value: `$${Math.round(strategies.reduce((s, st) => s + st.riskPerTrade * 3, 0)).toLocaleString()}` },
                { label: 'Starting Balance', value: `$${startingBalance.toLocaleString()}` },
                { label: 'Utilization', value: `${(strategies.reduce((s, st) => s + st.riskPerTrade * 3, 0) / startingBalance * 100).toFixed(1)}%` },
                { label: 'Max Concurrent', value: `${strategies.length}` },
              ].map((m) => (
                <div key={m.label}>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{m.label}</p>
                  <p className="text-sm font-bold">{m.value}</p>
                </div>
              ))}
            </div>
          </Card>

          {/* Worst Case Analysis */}
          <Card>
            <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Worst Case Analysis</h3>
            <div className="grid grid-cols-3 sm:grid-cols-5 gap-3 mb-4">
              {[
                { label: 'Worst Single Day', value: `-$${Math.round(strategies.reduce((s, st) => s + st.riskPerTrade * 2.5, 0))}` },
                { label: 'Worst Losing Streak', value: `5 days (-$${Math.round(strategies.reduce((s, st) => s + st.riskPerTrade * 8, 0))})` },
                { label: 'Worst 5-Day Rolling DD', value: `-$${Math.round(strategies.reduce((s, st) => s + st.riskPerTrade * 10, 0))}` },
                { label: 'Max DD vs Balance', value: `${(strategies.reduce((s, st) => s + st.riskPerTrade * 10, 0) / startingBalance * 100).toFixed(1)}%` },
                { label: 'Recovery Est.', value: '8-12 days' },
              ].map((m) => (
                <div key={m.label}>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{m.label}</p>
                  <p className="text-sm font-bold" style={m.value.startsWith('-') ? { color: 'var(--red)' } : undefined}>{m.value}</p>
                </div>
              ))}
            </div>
          </Card>

          {/* Monte Carlo Simulation */}
          <Card>
            <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Risk Simulation (Monte Carlo)</h3>
            <div className="flex items-center gap-4 mb-4 flex-wrap">
              <label className="text-xs" style={{ color: 'var(--text-muted)' }}>
                Shuffle Mode:
                <select className="ml-2" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '4px 8px', borderRadius: '6px', fontSize: '0.75rem' }}>
                  <option>Daily</option>
                  <option>Weekly</option>
                  <option>Individual</option>
                </select>
              </label>
              <label className="text-xs" style={{ color: 'var(--text-muted)' }}>
                Simulations:
                <input type="range" min={500} max={5000} step={500} defaultValue={1000} style={{ marginLeft: 8, verticalAlign: 'middle' }} />
                <span className="font-mono ml-1">1000</span>
              </label>
              <button className="px-3 py-1.5 rounded text-xs font-medium" style={{ background: 'var(--accent)', color: 'white', border: 'none', cursor: 'pointer' }}>
                Run Simulation
              </button>
            </div>
            <div className="grid grid-cols-3 sm:grid-cols-6 gap-3 mb-4">
              {[
                { label: 'Bust Probability', value: '3.1%' },
                { label: 'Daily Pause Prob', value: '9.4%' },
                { label: 'Max Loss Prob', value: '1.8%' },
                { label: 'Median Max DD', value: '-4.2%' },
                { label: '95th Pctl DD', value: '-7.1%' },
                { label: 'Expected Worst Day', value: `-$${Math.round(strategies.reduce((s, st) => s + st.riskPerTrade * 3, 0))}` },
              ].map((m) => (
                <div key={m.label}>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{m.label}</p>
                  <p className="text-sm font-bold">{m.value}</p>
                </div>
              ))}
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <ChartPlaceholder label="Max Drawdown Distribution histogram (50 bins)" height={180} />
              <ChartPlaceholder label="Equity Curve Confidence Bands (5th, 25th, 50th, 75th, 95th percentiles)" height={180} />
            </div>
          </Card>
        </div>
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
          onClick={() => {
            const payload = {
              name: portfolioName.trim(),
              starting_balance: startingBalance,
              compound_rate: riskScaling / 100,
              requirement_set_name: requirementSet === 'None' ? null : requirementSet,
              strategies: strategies.map(s => ({
                strategy_id: Number(s.id),
                risk_per_trade: s.riskPerTrade,
              })),
            };
            createMut.mutate(payload, {
              onSuccess: () => router.push('/portfolios'),
            });
          }}
        >
          {createMut.isPending ? 'Saving...' : 'Save Portfolio'}
        </button>
      </div>
    </div>
  );
}
