'use client';

import { useState } from 'react';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import Card from '@/components/Card';
import MetricCard from '@/components/MetricCard';
import ChartPlaceholder from '@/components/ChartPlaceholder';

/* =========================================================================
   MOCK DATA
   ========================================================================= */

// ---- Dashboard (combines Live Dashboard + Performance) ----
const dashKpis = {
  trades: 847,
  winRate: 55.2,
  pf: 1.89,
  totalPnL: 4230,
  balance: 84230,
  maxDD: -2.1,
};

const openPositions = [
  { strategy: 'SPY LONG - Mass #1', symbol: 'SPY', direction: 'LONG', entryPrice: 595.20, currentPrice: 596.85, unrealizedPnL: 82.50, duration: '1h 23m' },
  { strategy: 'NVDA LONG - Mass #2', symbol: 'NVDA', direction: 'LONG', entryPrice: 142.35, currentPrice: 141.80, unrealizedPnL: -27.50, duration: '42m' },
];

const strategyEquityData = [
  { name: 'SPY LONG - Mass #1', color: 'var(--accent)', data: [0, 200, 350, 520, 480, 650, 800, 920, 1050, 1180, 1350, 1280, 1420, 1550, 1680] },
  { name: 'NVDA LONG - Mass #2', color: 'var(--green)', data: [0, 150, 280, 200, 350, 500, 620, 710, 680, 800, 950, 1020, 1100, 1250, 1380] },
  { name: 'AAPL LONG - Mass #5', color: 'var(--orange)', data: [0, 80, 120, 200, 180, 310, 400, 350, 420, 480, 530, 600, 650, 720, 770] },
];

const combinedEquity = [0, 430, 750, 920, 1010, 1460, 1820, 1980, 2150, 2460, 2830, 2900, 3170, 3520, 3830];
const drawdownData = [0, -0.2, -0.5, -0.3, -1.2, -0.8, -0.4, -1.5, -2.1, -1.3, -0.6, -1.0, -0.4, -0.2, -0.1];

const liveTradeHistory = [
  { time: '10:32 AM', strategy: 'SPY LONG - Mass #1', symbol: 'SPY', entryPrice: 595.20, exitPrice: null, pnl: null, status: 'Open' },
  { time: '10:15 AM', strategy: 'NVDA LONG - Mass #2', symbol: 'NVDA', entryPrice: 142.35, exitPrice: null, pnl: null, status: 'Open' },
  { time: '09:58 AM', strategy: 'AAPL LONG - Mass #5', symbol: 'AAPL', entryPrice: 248.10, exitPrice: 249.30, pnl: 60.00, status: 'Closed' },
  { time: '09:45 AM', strategy: 'META LONG - Mass #13', symbol: 'META', entryPrice: 612.50, exitPrice: 611.20, pnl: -32.50, status: 'Closed' },
  { time: '09:38 AM', strategy: 'SPY LONG - Mass #3', symbol: 'SPY', entryPrice: 594.80, exitPrice: 595.65, pnl: 42.50, status: 'Closed' },
  { time: '09:31 AM', strategy: 'NVDA LONG - Mass #2', symbol: 'NVDA', entryPrice: 141.90, exitPrice: 141.30, pnl: -30.00, status: 'Closed' },
];

// ---- Strategies ----
const portfolioStrategies = [
  {
    id: '1', name: 'SPY LONG - Mass #1', symbol: 'SPY', direction: 'LONG',
    status: 'Healthy' as const, riskPerTrade: 100, winRate: 58.3, pf: 2.12, trades: 312, pnlContribution: 1680,
  },
  {
    id: '2', name: 'NVDA LONG - Mass #2', symbol: 'NVDA', direction: 'LONG',
    status: 'Warning' as const, riskPerTrade: 100, winRate: 54.0, pf: 2.05, trades: 224, pnlContribution: 1380,
  },
  {
    id: '3', name: 'AAPL LONG - Mass #5', symbol: 'AAPL', direction: 'LONG',
    status: 'Healthy' as const, riskPerTrade: 75, winRate: 52.8, pf: 1.78, trades: 186, pnlContribution: 770,
  },
  {
    id: '4', name: 'META LONG - Mass #13', symbol: 'META', direction: 'LONG',
    status: 'Critical' as const, riskPerTrade: 50, winRate: 45.5, pf: 1.22, trades: 66, pnlContribution: -120,
  },
  {
    id: '5', name: 'TSLA LONG - Mass #7', symbol: 'TSLA', direction: 'LONG',
    status: 'Healthy' as const, riskPerTrade: 80, winRate: 51.2, pf: 1.65, trades: 59, pnlContribution: 520,
  },
];

// ---- Compliance (combines Prop Firm + Account) ----
const requirementSet = {
  name: 'FTMO Challenge - $100K',
  status: 'Violations' as const,
};

const complianceRules = [
  { name: 'Maximum Daily Loss', threshold: '$5,000 (5%)', currentValue: '$1,240', currentPct: 24.8, passing: true },
  { name: 'Maximum Total Drawdown', threshold: '$10,000 (10%)', currentValue: '$2,180', currentPct: 21.8, passing: true },
  { name: 'Minimum Trading Days', threshold: '10 days', currentValue: '7 days', currentPct: 70, passing: false },
  { name: 'Profit Target', threshold: '$10,000 (10%)', currentValue: '$4,230', currentPct: 42.3, passing: false },
  { name: 'Maximum Position Size', threshold: '2% of balance', currentValue: '0.12%', currentPct: 6, passing: true },
  { name: 'No Weekend Holding', threshold: 'Close by Friday 4PM', currentValue: 'Compliant', currentPct: 100, passing: true },
];

const accountSummary = {
  currentBalance: 85161,
  startingBalance: 80000,
  netDeposits: 5000,
  tradingPnL: 161,
};

// ---- Settings (combines Webhooks + Deploy) ----
const webhookConfigs = [
  { strategy: 'SPY LONG - Mass #1', url: 'https://discord.com/api/webhooks/1234567890/abcdef', enabled: true },
  { strategy: 'NVDA LONG - Mass #2', url: 'https://hooks.slack.com/services/T01/B02/xyz', enabled: true },
  { strategy: 'AAPL LONG - Mass #5', url: '', enabled: false },
  { strategy: 'META LONG - Mass #13', url: 'https://discord.com/api/webhooks/9876543210/ghijkl', enabled: true },
  { strategy: 'TSLA LONG - Mass #7', url: '', enabled: false },
];

const deployStatus = {
  environment: 'Railway' as const,
  workerStatus: 'Running',
  lastDeploy: '2026-03-20 08:15 AM',
  uptime: '36h 17m',
};

/* =========================================================================
   HELPER COMPONENTS
   ========================================================================= */

// ---- SVG Mini Equity Curve ----
function MiniEquityCurve({ data, width = 320, height = 80, gradientId = 'eqGrad', color }: {
  data: number[];
  width?: number;
  height?: number;
  gradientId?: string;
  color?: string;
}) {
  if (data.length < 2) return null;

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const padding = 4;
  const chartH = height - padding * 2;
  const chartW = width - padding * 2;

  const points = data.map((v, i) => {
    const x = padding + (i / (data.length - 1)) * chartW;
    const y = padding + chartH - ((v - min) / range) * chartH;
    return `${x},${y}`;
  });

  const pathD = `M${points.join(' L')}`;
  const fillD = `${pathD} L${padding + chartW},${height} L${padding},${height} Z`;
  const finalVal = data[data.length - 1];
  const strokeColor = color || (finalVal >= 0 ? 'var(--green)' : 'var(--red)');

  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={strokeColor} stopOpacity={0.25} />
          <stop offset="100%" stopColor={strokeColor} stopOpacity={0} />
        </linearGradient>
      </defs>
      <path d={fillD} fill={`url(#${gradientId})`} />
      <path d={pathD} fill="none" stroke={strokeColor} strokeWidth="2" strokeLinejoin="round" />
    </svg>
  );
}

// ---- Progress Bar ----
function ProgressBar({ pct, color = 'var(--accent)', height = 8 }: {
  pct: number;
  color?: string;
  height?: number;
}) {
  const clampedPct = Math.min(100, Math.max(0, pct));
  return (
    <div className="w-full rounded-full overflow-hidden" style={{ background: 'var(--bg-input)', height }}>
      <div
        className="rounded-full transition-all"
        style={{ width: `${clampedPct}%`, height: '100%', background: color }}
      />
    </div>
  );
}

// ---- Status Badge ----
function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, { bg: string; text: string }> = {
    Healthy: { bg: 'var(--green-muted)', text: 'var(--green)' },
    Warning: { bg: 'var(--orange-muted)', text: 'var(--orange)' },
    Critical: { bg: 'var(--red-muted)', text: 'var(--red)' },
    Running: { bg: 'var(--green-muted)', text: 'var(--green)' },
  };
  const c = colors[status] || { bg: 'var(--accent-muted)', text: 'var(--accent)' };

  return (
    <span
      className="text-xs px-2 py-0.5 rounded-full font-medium"
      style={{ background: c.bg, color: c.text }}
    >
      {status}
    </span>
  );
}

/* =========================================================================
   TAB COMPONENTS
   ========================================================================= */

const TABS = ['Dashboard', 'Strategies', 'Compliance', 'Settings'];

// ---- 1. Dashboard (combines Live Dashboard + Performance) ----
function DashboardTab() {
  return (
    <div>
      {/* KPI Row */}
      <div className="grid grid-cols-6 gap-3 mb-6">
        <MetricCard label="Trades" value={String(dashKpis.trades)} />
        <MetricCard label="Win Rate" value={`${dashKpis.winRate}%`} />
        <MetricCard label="PF" value={String(dashKpis.pf)} />
        <MetricCard label="Total P&L" value={`$${dashKpis.totalPnL.toLocaleString()}`} delta="+5.3%" positive />
        <MetricCard label="Balance" value={`$${dashKpis.balance.toLocaleString()}`} />
        <MetricCard label="Max DD" value={`${dashKpis.maxDD}%`} />
      </div>

      {/* Open Positions (compact) */}
      {openPositions.length > 0 && (
        <Card className="mb-6">
          <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>
            Open Positions
            <span className="ml-2 text-xs px-1.5 py-0.5 rounded" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
              {openPositions.length}
            </span>
          </h3>

          <div className="grid grid-cols-7 gap-2 pb-2 border-b" style={{ borderColor: 'var(--border)' }}>
            <p className="col-span-2 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Strategy</p>
            <p className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Symbol</p>
            <p className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Entry</p>
            <p className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Current</p>
            <p className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Unrealized</p>
            <p className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Duration</p>
          </div>

          {openPositions.map((pos, i) => (
            <div key={i} className="grid grid-cols-7 gap-2 py-2.5 border-b items-center" style={{ borderColor: 'var(--border)' }}>
              <p className="col-span-2 text-sm font-medium">{pos.strategy}</p>
              <p className="text-sm">{pos.symbol}</p>
              <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>${pos.entryPrice.toFixed(2)}</p>
              <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>${pos.currentPrice.toFixed(2)}</p>
              <p className="text-sm font-medium" style={{ color: pos.unrealizedPnL >= 0 ? 'var(--green)' : 'var(--red)' }}>
                {pos.unrealizedPnL >= 0 ? '+' : ''}${pos.unrealizedPnL.toFixed(2)}
              </p>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{pos.duration}</p>
            </div>
          ))}
        </Card>
      )}

      {/* Combined Equity Curve */}
      <Card className="mb-6">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Combined Equity Curve</h3>
          <div className="flex items-center gap-4">
            {strategyEquityData.map((s) => (
              <span key={s.name} className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--text-muted)' }}>
                <span className="w-3 h-0.5 inline-block" style={{ background: s.color, opacity: 0.5 }} />
                {s.name.split(' - ')[0]}
              </span>
            ))}
            <span className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--text-muted)' }}>
              <span className="w-3 h-0.5 inline-block" style={{ background: 'var(--text-primary)' }} />
              Combined
            </span>
          </div>
        </div>

        <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)' }}>
          <div style={{ position: 'relative' }}>
            {strategyEquityData.map((s) => (
              <div key={s.name} style={{ position: 'absolute', top: 0, left: 0, right: 0, opacity: 0.4 }}>
                <MiniEquityCurve data={s.data} height={300} gradientId={`v3-perf-${s.name.replace(/\s/g, '')}`} color={s.color} />
              </div>
            ))}
            <div style={{ position: 'relative' }}>
              <MiniEquityCurve data={combinedEquity} height={300} gradientId="v3-perfCombined" color="var(--text-primary)" />
            </div>
          </div>
        </div>
      </Card>

      {/* Drawdown */}
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Drawdown</h3>
        <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)' }}>
          <MiniEquityCurve data={drawdownData} height={160} gradientId="v3-ddCurve" color="var(--red)" />
        </div>
        <div className="flex items-center gap-6 mt-3">
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Max Drawdown</p>
            <p className="text-sm font-semibold" style={{ color: 'var(--red)' }}>-2.1%</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Avg Drawdown</p>
            <p className="text-sm font-semibold">-0.74%</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Recovery Time</p>
            <p className="text-sm font-semibold">3.2 days</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Current DD</p>
            <p className="text-sm font-semibold" style={{ color: 'var(--green)' }}>-0.1%</p>
          </div>
        </div>
      </Card>

      {/* Trade History (compact) */}
      <Card>
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Recent Trades</h3>

        <div className="grid grid-cols-7 gap-2 pb-2 border-b" style={{ borderColor: 'var(--border)' }}>
          <p className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Time</p>
          <p className="col-span-2 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Strategy</p>
          <p className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Symbol</p>
          <p className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Entry</p>
          <p className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>P&L</p>
          <p className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Status</p>
        </div>

        <div style={{ maxHeight: 250, overflowY: 'auto' }}>
          {liveTradeHistory.map((trade, i) => (
            <div key={i} className="grid grid-cols-7 gap-2 py-2 border-b items-center" style={{ borderColor: 'var(--border)' }}>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{trade.time}</p>
              <p className="col-span-2 text-sm">{trade.strategy}</p>
              <p className="text-sm">{trade.symbol}</p>
              <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>${trade.entryPrice.toFixed(2)}</p>
              <p className="text-sm font-medium" style={{ color: trade.pnl === null ? 'var(--text-muted)' : trade.pnl >= 0 ? 'var(--green)' : 'var(--red)' }}>
                {trade.pnl === null ? '--' : `${trade.pnl >= 0 ? '+' : ''}$${trade.pnl.toFixed(2)}`}
              </p>
              <span
                className="text-xs px-1.5 py-0.5 rounded text-center"
                style={{
                  background: trade.status === 'Open' ? 'var(--accent-muted)' : 'var(--bg-input)',
                  color: trade.status === 'Open' ? 'var(--accent)' : 'var(--text-muted)',
                }}
              >
                {trade.status}
              </span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

// ---- 2. Strategies (simplified -- no R-distribution sparklines) ----
function StrategiesTab() {
  return (
    <div className="space-y-4">
      {portfolioStrategies.map((strat) => (
        <Card key={strat.id}>
          <div className="flex items-start justify-between mb-3">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <p className="font-medium">{strat.name}</p>
                <StatusBadge status={strat.status} />
              </div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                {strat.symbol} | {strat.direction} | ${strat.riskPerTrade}/trade
              </p>
            </div>
            <button className="px-3 py-1.5 rounded-lg text-xs font-medium" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>
              View Strategy
            </button>
          </div>

          {/* Strategy KPIs */}
          <div className="grid grid-cols-4 gap-4">
            <div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Win Rate</p>
              <p className="text-sm font-semibold">{strat.winRate}%</p>
            </div>
            <div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Profit Factor</p>
              <p className="text-sm font-semibold">{strat.pf.toFixed(2)}</p>
            </div>
            <div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Trades</p>
              <p className="text-sm font-semibold">{strat.trades}</p>
            </div>
            <div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>P&L Contribution</p>
              <p className="text-sm font-semibold" style={{ color: strat.pnlContribution >= 0 ? 'var(--green)' : 'var(--red)' }}>
                {strat.pnlContribution >= 0 ? '+' : ''}${strat.pnlContribution.toLocaleString()}
              </p>
            </div>
          </div>
        </Card>
      ))}
    </div>
  );
}

// ---- 3. Compliance (combines Prop Firm Check + Account balance summary) ----
function ComplianceTab() {
  const passingCount = complianceRules.filter((r) => r.passing).length;
  const totalRules = complianceRules.length;

  return (
    <div>
      {/* Requirement Set Header + Balance Summary */}
      <div className="grid grid-cols-2 gap-6 mb-6">
        <Card>
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold">{requirementSet.name}</h3>
              <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>
                {passingCount}/{totalRules} rules passing
              </p>
            </div>
            <span
              className="text-sm px-3 py-1 rounded-full font-medium"
              style={{
                background: requirementSet.status === 'Violations' ? 'var(--red-muted)' : 'var(--green-muted)',
                color: requirementSet.status === 'Violations' ? 'var(--red)' : 'var(--green)',
              }}
            >
              {requirementSet.status === 'Violations' ? 'Has Violations' : 'All Passing'}
            </span>
          </div>
        </Card>

        <Card>
          <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Balance Summary</h3>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Current Balance</p>
              <p className="text-lg font-semibold" style={{ color: 'var(--green)' }}>${accountSummary.currentBalance.toLocaleString()}</p>
            </div>
            <div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Starting Balance</p>
              <p className="text-lg font-semibold">${accountSummary.startingBalance.toLocaleString()}</p>
            </div>
            <div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Net Deposits</p>
              <p className="text-sm font-semibold">${accountSummary.netDeposits.toLocaleString()}</p>
            </div>
            <div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Trading P&L</p>
              <p className="text-sm font-semibold" style={{ color: 'var(--green)' }}>+${accountSummary.tradingPnL}</p>
            </div>
          </div>
        </Card>
      </div>

      {/* Compliance Rules */}
      <div className="space-y-3">
        {complianceRules.map((rule, i) => (
          <Card key={i}>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <p className="text-sm font-medium">{rule.name}</p>
                <span
                  className="text-xs px-2 py-0.5 rounded-full font-medium"
                  style={{
                    background: rule.passing ? 'var(--green-muted)' : 'var(--red-muted)',
                    color: rule.passing ? 'var(--green)' : 'var(--red)',
                  }}
                >
                  {rule.passing ? 'PASS' : 'FAIL'}
                </span>
              </div>
              <div className="flex items-center gap-3">
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Threshold: {rule.threshold}</p>
                <p className="text-sm font-semibold">{rule.currentValue}</p>
              </div>
            </div>

            <ProgressBar
              pct={rule.currentPct}
              color={rule.passing ? (rule.currentPct > 75 ? 'var(--orange)' : 'var(--green)') : 'var(--red)'}
              height={6}
            />
            <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
              {rule.currentPct.toFixed(1)}% of limit
            </p>
          </Card>
        ))}
      </div>
    </div>
  );
}

// ---- 4. Settings (combines Webhooks + Deploy) ----
function SettingsTab() {
  return (
    <div>
      {/* Deploy Status (compact) */}
      <Card className="mb-6">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Deployment</h3>
          <StatusBadge status={deployStatus.workerStatus} />
        </div>
        <div className="grid grid-cols-4 gap-4">
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Environment</p>
            <p className="text-sm font-semibold">{deployStatus.environment}</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Status</p>
            <p className="text-sm font-semibold" style={{ color: 'var(--green)' }}>{deployStatus.workerStatus}</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Last Deploy</p>
            <p className="text-sm font-semibold">{deployStatus.lastDeploy}</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Uptime</p>
            <p className="text-sm font-semibold">{deployStatus.uptime}</p>
          </div>
        </div>
        <div className="flex gap-2 mt-4">
          <button className="px-4 py-1.5 rounded-lg text-xs font-medium" style={{ background: 'var(--accent)', color: 'white' }}>
            Deploy
          </button>
          <button className="px-4 py-1.5 rounded-lg text-xs font-medium" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>
            Redeploy
          </button>
          <button className="px-4 py-1.5 rounded-lg text-xs font-medium" style={{ background: 'var(--red-muted)', color: 'var(--red)' }}>
            Stop Worker
          </button>
        </div>
      </Card>

      {/* Webhook Configs (compact table) */}
      <Card>
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Webhook Configuration</h3>

        <div className="grid grid-cols-12 gap-2 pb-2 border-b" style={{ borderColor: 'var(--border)' }}>
          <p className="col-span-3 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Strategy</p>
          <p className="col-span-5 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Webhook URL</p>
          <p className="col-span-2 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Status</p>
          <p className="col-span-2 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Actions</p>
        </div>

        {webhookConfigs.map((wh, i) => (
          <div key={i} className="grid grid-cols-12 gap-2 py-3 border-b items-center" style={{ borderColor: 'var(--border)' }}>
            <p className="col-span-3 text-sm font-medium">{wh.strategy}</p>
            <p className="col-span-5 text-xs font-mono truncate" style={{ color: wh.url ? 'var(--text-secondary)' : 'var(--text-muted)' }}>
              {wh.url || 'Not configured'}
            </p>
            <div className="col-span-2">
              <span
                className="text-xs px-2 py-0.5 rounded-full"
                style={{
                  background: wh.enabled ? 'var(--green-muted)' : 'rgba(156,163,175,0.15)',
                  color: wh.enabled ? 'var(--green)' : 'var(--text-muted)',
                }}
              >
                {wh.enabled ? 'Enabled' : 'Disabled'}
              </span>
            </div>
            <div className="col-span-2 flex gap-1">
              <button className="px-2 py-1 rounded text-xs" style={{ background: 'var(--bg-input)', color: 'var(--text-secondary)' }}>
                Edit
              </button>
              {wh.enabled && (
                <button className="px-2 py-1 rounded text-xs" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                  Test
                </button>
              )}
            </div>
          </div>
        ))}
      </Card>
    </div>
  );
}

/* =========================================================================
   MAIN V3 COMPONENT
   ========================================================================= */

export default function PortfolioDetailV3() {
  return (
    <div>
      <PageHeader
        title="My Portfolio"
        subtitle="5 strategies | $80,000 starting balance"
        backHref="/portfolios"
        actions={
          <>
            <button className="px-3 py-1.5 rounded-lg text-sm" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>Edit</button>
            <button className="px-3 py-1.5 rounded-lg text-sm" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>Clone</button>
          </>
        }
      />

      <TabBar tabs={TABS}>
        {(tab) => (
          <div>
            {tab === 'Dashboard' && <DashboardTab />}
            {tab === 'Strategies' && <StrategiesTab />}
            {tab === 'Compliance' && <ComplianceTab />}
            {tab === 'Settings' && <SettingsTab />}
          </div>
        )}
      </TabBar>
    </div>
  );
}
