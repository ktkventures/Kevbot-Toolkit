'use client';

import { useState } from 'react';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import Card from '@/components/Card';
import Modal from '@/components/Modal';
import MetricCard from '@/components/MetricCard';
import ChartPlaceholder from '@/components/ChartPlaceholder';

/* =========================================================================
   MOCK DATA
   ========================================================================= */

// ---- Live Dashboard ----
const liveDashKpis = {
  alertTrades: 17,
  winRate: 52.9,
  totalPnL: 161,
  expectedPnL: 305,
  vsPlan: -144,
};

const openPositions = [
  { strategy: 'SPY LONG - Mass #1', symbol: 'SPY', direction: 'LONG', entryPrice: 595.20, currentPrice: 596.85, unrealizedPnL: 82.50, duration: '1h 23m' },
  { strategy: 'NVDA LONG - Mass #2', symbol: 'NVDA', direction: 'LONG', entryPrice: 142.35, currentPrice: 141.80, unrealizedPnL: -27.50, duration: '42m' },
];

const buyingPower = {
  startingBalance: 80000,
  currentBalance: 85161,
  allocated: 12500,
  available: 72661,
  utilization: 14.7,
};

const anomalies = [
  { severity: 'warning' as const, message: '3 consecutive losses on SPY LONG - Mass #1', time: '10:45 AM' },
  { severity: 'critical' as const, message: 'Drawdown exceeds 2x expected on NVDA LONG - Mass #2', time: '10:22 AM' },
  { severity: 'info' as const, message: 'Win rate trending below benchmark for META LONG - Mass #13', time: '09:55 AM' },
];

const liveTradeHistory = [
  { time: '10:32 AM', strategy: 'SPY LONG - Mass #1', symbol: 'SPY', direction: 'LONG', entryPrice: 595.20, exitPrice: null, pnl: null, status: 'Open' },
  { time: '10:15 AM', strategy: 'NVDA LONG - Mass #2', symbol: 'NVDA', direction: 'LONG', entryPrice: 142.35, exitPrice: null, pnl: null, status: 'Open' },
  { time: '09:58 AM', strategy: 'AAPL LONG - Mass #5', symbol: 'AAPL', direction: 'LONG', entryPrice: 248.10, exitPrice: 249.30, pnl: 60.00, status: 'Closed' },
  { time: '09:45 AM', strategy: 'META LONG - Mass #13', symbol: 'META', direction: 'LONG', entryPrice: 612.50, exitPrice: 611.20, pnl: -32.50, status: 'Closed' },
  { time: '09:38 AM', strategy: 'SPY LONG - Mass #3', symbol: 'SPY', direction: 'LONG', entryPrice: 594.80, exitPrice: 595.65, pnl: 42.50, status: 'Closed' },
  { time: '09:31 AM', strategy: 'NVDA LONG - Mass #2', symbol: 'NVDA', direction: 'LONG', entryPrice: 141.90, exitPrice: 141.30, pnl: -30.00, status: 'Closed' },
  { time: 'Yesterday', strategy: 'SPY LONG - Mass #1', symbol: 'SPY', direction: 'LONG', entryPrice: 593.50, exitPrice: 594.80, pnl: 65.00, status: 'Closed' },
  { time: 'Yesterday', strategy: 'AAPL LONG - Mass #5', symbol: 'AAPL', direction: 'LONG', entryPrice: 247.20, exitPrice: 246.50, pnl: -35.00, status: 'Closed' },
];

// ---- Performance ----
const perfKpis = {
  trades: 847,
  winRate: 55.2,
  pf: 1.89,
  totalPnL: 4230,
  balance: 84230,
  maxDD: -2.1,
};

const strategyEquityData = [
  { name: 'SPY LONG - Mass #1', color: 'var(--accent)', data: [0, 200, 350, 520, 480, 650, 800, 920, 1050, 1180, 1350, 1280, 1420, 1550, 1680] },
  { name: 'NVDA LONG - Mass #2', color: 'var(--green)', data: [0, 150, 280, 200, 350, 500, 620, 710, 680, 800, 950, 1020, 1100, 1250, 1380] },
  { name: 'AAPL LONG - Mass #5', color: 'var(--orange)', data: [0, 80, 120, 200, 180, 310, 400, 350, 420, 480, 530, 600, 650, 720, 770] },
];

const combinedEquity = [0, 430, 750, 920, 1010, 1460, 1820, 1980, 2150, 2460, 2830, 2900, 3170, 3520, 3830];

const drawdownData = [0, -0.2, -0.5, -0.3, -1.2, -0.8, -0.4, -1.5, -2.1, -1.3, -0.6, -1.0, -0.4, -0.2, -0.1];

// ---- Strategies ----
const portfolioStrategies = [
  {
    id: '1', name: 'SPY LONG - Mass #1', symbol: 'SPY', direction: 'LONG',
    status: 'Healthy' as const, healthScore: 92,
    riskPerTrade: 100, winRate: 58.3, pf: 2.12, trades: 312, pnlContribution: 1680,
    rDistribution: [1.2, -0.8, 2.1, 0.5, -0.3, 1.8, -0.6, 1.5, 2.3, -0.9, 0.7, 1.1, -0.4, 1.9, 0.3, -0.7, 2.5, 1.0, -0.5, 1.6],
  },
  {
    id: '2', name: 'NVDA LONG - Mass #2', symbol: 'NVDA', direction: 'LONG',
    status: 'Warning' as const, healthScore: 68,
    riskPerTrade: 100, winRate: 54.0, pf: 2.05, trades: 224, pnlContribution: 1380,
    rDistribution: [0.8, -1.2, 1.5, -0.6, 0.3, 2.0, -0.9, 1.1, -0.4, 1.7, 0.5, -0.8, 1.3, -1.0, 2.2, 0.6, -0.3, 1.8, -0.7, 0.9],
  },
  {
    id: '3', name: 'AAPL LONG - Mass #5', symbol: 'AAPL', direction: 'LONG',
    status: 'Healthy' as const, healthScore: 85,
    riskPerTrade: 75, winRate: 52.8, pf: 1.78, trades: 186, pnlContribution: 770,
    rDistribution: [0.5, -0.6, 1.0, -0.3, 0.8, 1.2, -0.9, 0.4, -0.5, 1.5, 0.7, -0.4, 1.1, 0.3, -0.8, 0.6, 1.3, -0.7, 0.9, -0.2],
  },
  {
    id: '4', name: 'META LONG - Mass #13', symbol: 'META', direction: 'LONG',
    status: 'Critical' as const, healthScore: 38,
    riskPerTrade: 50, winRate: 45.5, pf: 1.22, trades: 66, pnlContribution: -120,
    rDistribution: [-0.8, 0.5, -1.1, -0.3, 0.9, -0.6, 0.2, -1.0, 0.4, -0.7, 0.3, -0.5, -0.9, 0.6, -0.4, 0.1, -0.8, 0.7, -0.6, -0.3],
  },
  {
    id: '5', name: 'TSLA LONG - Mass #7', symbol: 'TSLA', direction: 'LONG',
    status: 'Healthy' as const, healthScore: 78,
    riskPerTrade: 80, winRate: 51.2, pf: 1.65, trades: 59, pnlContribution: 520,
    rDistribution: [1.0, -0.5, 0.8, -0.7, 1.5, 0.3, -0.4, 1.2, -0.9, 0.6, 1.8, -0.3, 0.5, -0.6, 1.1, 0.4, -0.8, 1.4, 0.2, -0.5],
  },
];

// ---- Prop Firm Check ----
const requirementSet = {
  name: 'FTMO Challenge - $100K',
  status: 'Violations' as const,
};

const complianceRules = [
  {
    name: 'Maximum Daily Loss', type: 'Daily', threshold: '$5,000 (5%)', currentValue: '$1,240',
    currentPct: 24.8, passing: true,
    violations: [],
  },
  {
    name: 'Maximum Total Drawdown', type: 'Overall', threshold: '$10,000 (10%)', currentValue: '$2,180',
    currentPct: 21.8, passing: true,
    violations: [],
  },
  {
    name: 'Minimum Trading Days', type: 'Calendar', threshold: '10 days', currentValue: '7 days',
    currentPct: 70, passing: false,
    violations: ['Need 3 more trading days before evaluation deadline (2026-03-28)'],
  },
  {
    name: 'Profit Target', type: 'Overall', threshold: '$10,000 (10%)', currentValue: '$4,230',
    currentPct: 42.3, passing: false,
    violations: ['$5,770 remaining to reach profit target'],
  },
  {
    name: 'Maximum Position Size', type: 'Per Trade', threshold: '2% of balance', currentValue: '0.12%',
    currentPct: 6, passing: true,
    violations: [],
  },
  {
    name: 'No Weekend Holding', type: 'Session', threshold: 'All positions closed by Friday 4PM', currentValue: 'Compliant',
    currentPct: 100, passing: true,
    violations: [],
  },
];

const dailyLimitTracker = {
  limit: 5000,
  used: 1240,
  remaining: 3760,
  worstCaseLoss: 800,
  pctUsed: 24.8,
};

// ---- Account ----
const accountMetrics = {
  currentBalance: 85161,
  startingBalance: 80000,
  netDeposits: 5000,
  tradingPnL: 161,
};

const mockLedger = [
  {
    date: '2026-03-20', type: 'Daily Trading P&L', amount: 47.50, summary: '5 trades, 1 change',
    trades: [
      { note: 'SPY LONG - Mass #3 trade #12', amount: 32.00 },
      { note: 'NVDA LONG - Mass #2 trade #45', amount: -18.50 },
      { note: 'AAPL LONG - Mass #5 trade #22', amount: 15.00 },
      { note: 'SPY LONG - Mass #1 trade #8', amount: 28.00 },
      { note: 'META LONG - Mass #13 trade #3', amount: -9.00 },
    ],
    changes: ['Risk adjusted: SPY LONG Mass #1 $100 -> $115/trade'],
    journal: { mood: 'Focused', confidence: 4, notes: 'Good discipline today. Avoided overtrading during lunch chop.' },
  },
  {
    date: '2026-03-19', type: 'Daily Trading P&L', amount: 113.60, summary: '13 trades, has notes',
    trades: [
      { note: 'SPY LONG - Mass #1 trade #7', amount: 65.00 },
      { note: 'NVDA LONG - Mass #2 trade #44', amount: 42.30 },
      { note: 'AAPL LONG - Mass #5 trade #21', amount: -18.20 },
      { note: 'META LONG - Mass #13 trade #2', amount: 24.50 },
    ],
    changes: [],
    journal: { mood: 'Confident', confidence: 5, notes: 'Strong trend day. All strategies aligned. Let winners run.' },
  },
  {
    date: '2026-03-18', type: 'Deposit', amount: 5000.00, summary: 'Initial funding',
    trades: [], changes: [],
    journal: null,
  },
  {
    date: '2026-03-17', type: 'Daily Trading P&L', amount: -32.40, summary: '8 trades',
    trades: [
      { note: 'SPY LONG - Mass #1 trade #6', amount: -45.00 },
      { note: 'NVDA LONG - Mass #2 trade #43', amount: 12.60 },
    ],
    changes: [],
    journal: { mood: 'Frustrated', confidence: 2, notes: 'Choppy day. SPY whipsawed multiple times. Need better session filter.' },
  },
  {
    date: '2026-03-14', type: 'Daily Trading P&L', amount: 85.20, summary: '11 trades',
    trades: [
      { note: 'Multiple trades across 5 strategies', amount: 85.20 },
    ],
    changes: ['Added TSLA LONG - Mass #7 to portfolio'],
    journal: { mood: 'Calm', confidence: 4, notes: 'Steady gains. New TSLA strategy showing promise.' },
  },
];

// ---- Webhooks ----
const webhookConfigs = [
  { strategy: 'SPY LONG - Mass #1', url: 'https://discord.com/api/webhooks/1234567890/abcdef', template: 'Discord Rich Embed', enabled: true },
  { strategy: 'NVDA LONG - Mass #2', url: 'https://hooks.slack.com/services/T01/B02/xyz', template: 'Slack Block Kit', enabled: true },
  { strategy: 'AAPL LONG - Mass #5', url: '', template: 'None', enabled: false },
  { strategy: 'META LONG - Mass #13', url: 'https://discord.com/api/webhooks/9876543210/ghijkl', template: 'Discord Rich Embed', enabled: true },
  { strategy: 'TSLA LONG - Mass #7', url: '', template: 'None', enabled: false },
];

const webhookDeliveries = [
  { time: '10:32 AM', strategy: 'SPY LONG - Mass #1', event: 'Entry Signal', status: 'success' as const, responseCode: 204, latency: '120ms' },
  { time: '10:15 AM', strategy: 'NVDA LONG - Mass #2', event: 'Entry Signal', status: 'success' as const, responseCode: 200, latency: '85ms' },
  { time: '09:58 AM', strategy: 'AAPL LONG - Mass #5', event: 'Exit Signal', status: 'failed' as const, responseCode: 0, latency: 'timeout' },
  { time: '09:58 AM', strategy: 'AAPL LONG - Mass #5', event: 'Exit Signal (retry 1)', status: 'failed' as const, responseCode: 0, latency: 'timeout' },
  { time: '09:45 AM', strategy: 'META LONG - Mass #13', event: 'Entry Signal', status: 'success' as const, responseCode: 204, latency: '95ms' },
  { time: '09:38 AM', strategy: 'SPY LONG - Mass #3', event: 'Exit Signal', status: 'success' as const, responseCode: 204, latency: '110ms' },
  { time: 'Yesterday', strategy: 'SPY LONG - Mass #1', event: 'Entry Signal', status: 'success' as const, responseCode: 204, latency: '130ms' },
  { time: 'Yesterday', strategy: 'NVDA LONG - Mass #2', event: 'Exit Signal', status: 'success' as const, responseCode: 200, latency: '78ms' },
];

// ---- Deploy ----
const deployStatus = {
  environment: 'Railway' as const,
  workerStatus: 'Running',
  apiKeysConfigured: true,
  lastDeploy: '2026-03-20 08:15 AM',
  uptime: '36h 17m',
};

const monitoringStrategies = [
  { name: 'SPY LONG - Mass #1', enabled: true, lastSignal: '10:32 AM', status: 'Streaming' },
  { name: 'NVDA LONG - Mass #2', enabled: true, lastSignal: '10:15 AM', status: 'Streaming' },
  { name: 'AAPL LONG - Mass #5', enabled: true, lastSignal: '09:58 AM', status: 'Streaming' },
  { name: 'META LONG - Mass #13', enabled: false, lastSignal: 'N/A', status: 'Disabled' },
  { name: 'TSLA LONG - Mass #7', enabled: true, lastSignal: '09:35 AM', status: 'Streaming' },
];

const recentLogs = [
  { time: '10:32:15', level: 'INFO', message: '[SPY] Entry signal fired — [C] EMA Bull Cross, all confluence met' },
  { time: '10:15:03', level: 'INFO', message: '[NVDA] Entry signal fired — [C] EMA Bull Cross, RVOL HIGH' },
  { time: '09:58:44', level: 'INFO', message: '[AAPL] Exit signal — +1.2R, target reached' },
  { time: '09:45:22', level: 'WARN', message: '[META] Webhook delivery failed — timeout after 5s, retry queued' },
  { time: '09:38:11', level: 'INFO', message: '[SPY] Exit signal — +0.85R, bar count exit' },
  { time: '09:31:00', level: 'INFO', message: 'RTH session started — monitoring 4 strategies on 4 symbols' },
  { time: '09:30:05', level: 'INFO', message: 'WebSocket connected to Polygon.io — SIP feed' },
  { time: '09:30:00', level: 'INFO', message: 'Ralph Engine started — worker PID 48291' },
];

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

// ---- R-Distribution Sparkline ----
function RDistributionSparkline({ data, width = 200, height = 40 }: {
  data: number[];
  width?: number;
  height?: number;
}) {
  const max = Math.max(...data.map(Math.abs)) || 1;
  const barW = (width - data.length) / data.length;
  const mid = height / 2;

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      {data.map((v, i) => {
        const barH = (Math.abs(v) / max) * (height / 2 - 2);
        const y = v >= 0 ? mid - barH : mid;
        return (
          <rect
            key={i}
            x={i * (barW + 1)}
            y={y}
            width={barW}
            height={barH}
            fill={v >= 0 ? 'var(--green)' : 'var(--red)'}
            opacity={0.8}
            rx={1}
          />
        );
      })}
      <line x1={0} y1={mid} x2={width} y2={mid} stroke="var(--border)" strokeWidth={0.5} />
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
function StatusBadge({ status }: { status: 'Healthy' | 'Warning' | 'Critical' | string }) {
  const colors: Record<string, { bg: string; text: string }> = {
    Healthy: { bg: 'var(--green-muted)', text: 'var(--green)' },
    Warning: { bg: 'var(--orange-muted)', text: 'var(--orange)' },
    Critical: { bg: 'var(--red-muted)', text: 'var(--red)' },
    Streaming: { bg: 'var(--green-muted)', text: 'var(--green)' },
    Disabled: { bg: 'rgba(156,163,175,0.15)', text: 'var(--text-muted)' },
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

// ---- Confidence Stars ----
function ConfidenceStars({ rating }: { rating: number }) {
  return (
    <span className="inline-flex gap-0.5">
      {[1, 2, 3, 4, 5].map((star) => (
        <span
          key={star}
          className="text-sm"
          style={{ color: star <= rating ? 'var(--orange)' : 'var(--border)' }}
        >
          *
        </span>
      ))}
    </span>
  );
}

/* =========================================================================
   TAB COMPONENTS
   ========================================================================= */

const TABS = [
  'Live Dashboard', 'Performance', 'Strategies', 'Prop Firm Check',
  'Account', 'Webhooks', 'Deploy',
];

// ---- 1. Live Dashboard ----
function LiveDashboardTab() {
  return (
    <div>
      {/* KPI Row */}
      <div className="grid grid-cols-5 gap-3 mb-6">
        <MetricCard label="Alert Trades" value={String(liveDashKpis.alertTrades)} />
        <MetricCard label="Win Rate" value={`${liveDashKpis.winRate}%`} />
        <MetricCard label="Total P&L" value={`$${liveDashKpis.totalPnL}`} delta={`+$${liveDashKpis.totalPnL}`} positive />
        <MetricCard label="Expected P&L" value={`$${liveDashKpis.expectedPnL}`} delta="benchmark" />
        <MetricCard label="vs Plan" value={`-$${Math.abs(liveDashKpis.vsPlan)}`} delta="below plan" positive={false} />
      </div>

      {/* Performance vs Plan Chart */}
      <Card className="mb-6">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Performance vs Plan</h3>
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--text-muted)' }}>
              <span className="w-3 h-0.5 inline-block" style={{ background: 'var(--accent)', opacity: 0.5 }} /> Benchmark
            </span>
            <span className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--text-muted)' }}>
              <span className="w-3 h-0.5 inline-block rounded" style={{ background: 'rgba(99,102,241,0.2)' }} /> Confidence Band
            </span>
            <span className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--text-muted)' }}>
              <span className="w-3 h-0.5 inline-block" style={{ background: 'var(--green)' }} /> Actual
            </span>
          </div>
        </div>
        <ChartPlaceholder label="Benchmark equity line with +/- 1 std confidence bands overlaid with actual equity curve" height={400} />
      </Card>

      {/* Open Positions */}
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
          Open Positions
          <span className="ml-2 text-xs px-1.5 py-0.5 rounded" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
            {openPositions.length}
          </span>
        </h3>

        {/* Header */}
        <div className="grid grid-cols-12 gap-2 pb-2 border-b" style={{ borderColor: 'var(--border)' }}>
          {['Strategy', 'Symbol', 'Dir', 'Entry', 'Current', 'Unrealized P&L', 'Duration'].map((h, i) => (
            <p key={h} className={`text-xs font-medium ${i < 2 ? 'col-span-3' : i === 2 ? 'col-span-1' : 'col-span-1'}`}
               style={{ color: 'var(--text-muted)', gridColumn: i === 0 ? 'span 3' : i === 1 ? 'span 1' : 'span 1' }}>
              {h}
            </p>
          ))}
        </div>
        {/* Use a simpler header layout */}
        <div className="grid grid-cols-7 gap-2 pb-2 border-b -mt-8 pt-2" style={{ borderColor: 'var(--border)' }}>
          <p className="col-span-2 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Strategy</p>
          <p className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Symbol</p>
          <p className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Entry</p>
          <p className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Current</p>
          <p className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Unrealized</p>
          <p className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Duration</p>
        </div>

        {openPositions.map((pos, i) => (
          <div key={i} className="grid grid-cols-7 gap-2 py-3 border-b items-center" style={{ borderColor: 'var(--border)' }}>
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

      {/* Buying Power Tracker */}
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Buying Power Tracker</h3>
        <div className="grid grid-cols-4 gap-4 mb-4">
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Starting Balance</p>
            <p className="text-lg font-semibold">${buyingPower.startingBalance.toLocaleString()}</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Current Balance</p>
            <p className="text-lg font-semibold" style={{ color: 'var(--green)' }}>${buyingPower.currentBalance.toLocaleString()}</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Allocated</p>
            <p className="text-lg font-semibold" style={{ color: 'var(--orange)' }}>${buyingPower.allocated.toLocaleString()}</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Available</p>
            <p className="text-lg font-semibold">${buyingPower.available.toLocaleString()}</p>
          </div>
        </div>
        <div>
          <div className="flex items-center justify-between mb-1">
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Capital Utilization</p>
            <p className="text-xs font-medium">{buyingPower.utilization}%</p>
          </div>
          <ProgressBar pct={buyingPower.utilization} color="var(--accent)" height={10} />
        </div>
      </Card>

      {/* Anomaly Detection */}
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>
          Anomaly Detection
          {anomalies.length > 0 && (
            <span className="ml-2 text-xs px-1.5 py-0.5 rounded" style={{ background: 'var(--red-muted)', color: 'var(--red)' }}>
              {anomalies.length}
            </span>
          )}
        </h3>
        <div className="space-y-2">
          {anomalies.map((anomaly, i) => {
            const severityColors = {
              critical: { bg: 'var(--red-muted)', text: 'var(--red)', label: 'CRITICAL' },
              warning: { bg: 'var(--orange-muted)', text: 'var(--orange)', label: 'WARNING' },
              info: { bg: 'var(--accent-muted)', text: 'var(--accent)', label: 'INFO' },
            };
            const sc = severityColors[anomaly.severity];
            return (
              <div key={i} className="flex items-center gap-3 p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                <span className="text-xs px-2 py-0.5 rounded font-mono font-medium" style={{ background: sc.bg, color: sc.text }}>
                  {sc.label}
                </span>
                <p className="text-sm flex-1" style={{ color: 'var(--text-secondary)' }}>{anomaly.message}</p>
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{anomaly.time}</span>
              </div>
            );
          })}
        </div>
      </Card>

      {/* Trade History */}
      <Card>
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Trade History</h3>

        <div className="grid grid-cols-8 gap-2 pb-2 border-b" style={{ borderColor: 'var(--border)' }}>
          <p className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Time</p>
          <p className="col-span-2 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Strategy</p>
          <p className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Symbol</p>
          <p className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Entry</p>
          <p className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Exit</p>
          <p className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>P&L</p>
          <p className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Status</p>
        </div>

        <div style={{ maxHeight: 300, overflowY: 'auto' }}>
          {liveTradeHistory.map((trade, i) => (
            <div key={i} className="grid grid-cols-8 gap-2 py-2.5 border-b items-center" style={{ borderColor: 'var(--border)' }}>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{trade.time}</p>
              <p className="col-span-2 text-sm">{trade.strategy}</p>
              <p className="text-sm">{trade.symbol}</p>
              <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>${trade.entryPrice.toFixed(2)}</p>
              <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                {trade.exitPrice ? `$${trade.exitPrice.toFixed(2)}` : '--'}
              </p>
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

// ---- 2. Performance ----
function PerformanceTab() {
  return (
    <div>
      {/* KPI Row */}
      <div className="grid grid-cols-6 gap-3 mb-6">
        <MetricCard label="Trades" value={String(perfKpis.trades)} />
        <MetricCard label="Win Rate" value={`${perfKpis.winRate}%`} />
        <MetricCard label="PF" value={String(perfKpis.pf)} />
        <MetricCard label="Total P&L" value={`$${perfKpis.totalPnL.toLocaleString()}`} delta="+5.3%" positive />
        <MetricCard label="Balance" value={`$${perfKpis.balance.toLocaleString()}`} />
        <MetricCard label="Max DD" value={`${perfKpis.maxDD}%`} delta="controlled" />
      </div>

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
          {/* Per-strategy dotted lines layered under combined solid */}
          <div style={{ position: 'relative' }}>
            {strategyEquityData.map((s) => (
              <div key={s.name} style={{ position: 'absolute', top: 0, left: 0, right: 0, opacity: 0.4 }}>
                <MiniEquityCurve data={s.data} height={300} gradientId={`perf-${s.name.replace(/\s/g, '')}`} color={s.color} />
              </div>
            ))}
            <div style={{ position: 'relative' }}>
              <MiniEquityCurve data={combinedEquity} height={300} gradientId="perfCombined" color="var(--text-primary)" />
            </div>
          </div>
        </div>
      </Card>

      {/* Drawdown Analysis */}
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Drawdown Analysis</h3>
        <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)' }}>
          <MiniEquityCurve data={drawdownData} height={200} gradientId="perfDD" color="var(--red)" />
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

      {/* Daily P&L Distribution + Strategy Correlation Heatmap */}
      <div className="grid grid-cols-2 gap-6">
        <Card>
          <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Daily P&L Distribution</h3>
          <ChartPlaceholder label="Histogram: daily P&L distribution with mean and std lines" height={300} />
          <div className="flex items-center gap-6 mt-3">
            <div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Avg Daily</p>
              <p className="text-sm font-semibold" style={{ color: 'var(--green)' }}>+$142</p>
            </div>
            <div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Std Dev</p>
              <p className="text-sm font-semibold">$85</p>
            </div>
            <div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Best Day</p>
              <p className="text-sm font-semibold" style={{ color: 'var(--green)' }}>+$420</p>
            </div>
            <div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Worst Day</p>
              <p className="text-sm font-semibold" style={{ color: 'var(--red)' }}>-$185</p>
            </div>
          </div>
        </Card>

        <Card>
          <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Strategy Correlation Heatmap</h3>
          {/* Inline SVG heatmap */}
          <div className="rounded-lg p-4" style={{ background: 'var(--bg-input)' }}>
            <div className="overflow-auto">
              <table className="w-full text-xs" style={{ borderCollapse: 'separate', borderSpacing: 2 }}>
                <thead>
                  <tr>
                    <th />
                    {['SPY', 'NVDA', 'AAPL', 'META', 'TSLA'].map((s) => (
                      <th key={s} className="px-2 py-1 font-medium text-center" style={{ color: 'var(--text-muted)' }}>{s}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[
                    { label: 'SPY', values: [1.00, 0.42, 0.38, 0.15, 0.28] },
                    { label: 'NVDA', values: [0.42, 1.00, 0.31, 0.22, 0.55] },
                    { label: 'AAPL', values: [0.38, 0.31, 1.00, 0.18, 0.25] },
                    { label: 'META', values: [0.15, 0.22, 0.18, 1.00, 0.20] },
                    { label: 'TSLA', values: [0.28, 0.55, 0.25, 0.20, 1.00] },
                  ].map((row) => (
                    <tr key={row.label}>
                      <td className="px-2 py-1 font-medium" style={{ color: 'var(--text-muted)' }}>{row.label}</td>
                      {row.values.map((v, j) => {
                        const intensity = Math.abs(v);
                        const bg = v === 1 ? 'var(--accent)' : `rgba(99, 102, 241, ${intensity * 0.6})`;
                        return (
                          <td key={j} className="px-2 py-2 text-center rounded" style={{ background: bg, color: 'white', fontWeight: 500 }}>
                            {v.toFixed(2)}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}

// ---- 3. Strategies ----
function StrategiesTab() {
  return (
    <div className="space-y-4">
      {portfolioStrategies.map((strat) => (
        <Card key={strat.id}>
          <div className="flex items-start justify-between mb-3">
            <div className="flex items-center gap-3">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <p className="font-medium">{strat.name}</p>
                  <StatusBadge status={strat.status} />
                  <span className="text-xs px-1.5 py-0.5 rounded" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>
                    Score: {strat.healthScore}
                  </span>
                </div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  {strat.symbol} | {strat.direction} | ${strat.riskPerTrade}/trade
                </p>
              </div>
            </div>
            <div className="flex gap-2">
              <button className="px-3 py-1.5 rounded-lg text-xs font-medium" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>
                View Strategy
              </button>
              <button className="px-3 py-1.5 rounded-lg text-xs font-medium" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>
                View Chart
              </button>
            </div>
          </div>

          {/* Strategy KPIs */}
          <div className="grid grid-cols-4 gap-4 mb-3">
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

          {/* R-Distribution Sparkline */}
          <div className="rounded-lg p-2" style={{ background: 'var(--bg-input)' }}>
            <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>R-Distribution (last 20 trades)</p>
            <RDistributionSparkline data={strat.rDistribution} width={600} height={36} />
          </div>
        </Card>
      ))}
    </div>
  );
}

// ---- 4. Prop Firm Check ----
function PropFirmCheckTab() {
  const passingCount = complianceRules.filter((r) => r.passing).length;
  const totalRules = complianceRules.length;

  return (
    <div>
      {/* Header */}
      <Card className="mb-6">
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

      {/* Compliance Rules */}
      <div className="space-y-4 mb-6">
        {complianceRules.map((rule, i) => (
          <Card key={i}>
            <div className="flex items-start justify-between mb-3">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <p className="font-medium">{rule.name}</p>
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
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  Type: {rule.type} | Threshold: {rule.threshold}
                </p>
              </div>
              <div className="text-right">
                <p className="text-sm font-semibold">{rule.currentValue}</p>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>current</p>
              </div>
            </div>

            {/* Progress Bar */}
            <div className="mb-2">
              <div className="flex items-center justify-between mb-1">
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  {rule.currentPct.toFixed(1)}% of limit
                </p>
              </div>
              <ProgressBar
                pct={rule.currentPct}
                color={rule.passing ? (rule.currentPct > 75 ? 'var(--orange)' : 'var(--green)') : 'var(--red)'}
                height={8}
              />
            </div>

            {/* Violation History */}
            {rule.violations.length > 0 && (
              <div className="mt-3 pt-3 border-t" style={{ borderColor: 'var(--border)' }}>
                {rule.violations.map((violation, vi) => (
                  <div key={vi} className="flex items-center gap-2 py-1">
                    <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: 'var(--red)' }} />
                    <p className="text-xs" style={{ color: 'var(--red)' }}>{violation}</p>
                  </div>
                ))}
              </div>
            )}
          </Card>
        ))}
      </div>

      {/* Daily Limit Tracker */}
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Daily Loss Limit Tracker</h3>
        <div className="grid grid-cols-4 gap-4 mb-4">
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Daily Limit</p>
            <p className="text-lg font-semibold">${dailyLimitTracker.limit.toLocaleString()}</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Used Today</p>
            <p className="text-lg font-semibold" style={{ color: 'var(--orange)' }}>${dailyLimitTracker.used.toLocaleString()}</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Remaining</p>
            <p className="text-lg font-semibold" style={{ color: 'var(--green)' }}>${dailyLimitTracker.remaining.toLocaleString()}</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Worst Case (open)</p>
            <p className="text-lg font-semibold" style={{ color: 'var(--red)' }}>-${dailyLimitTracker.worstCaseLoss.toLocaleString()}</p>
          </div>
        </div>
        <ProgressBar pct={dailyLimitTracker.pctUsed} color={dailyLimitTracker.pctUsed > 75 ? 'var(--red)' : dailyLimitTracker.pctUsed > 50 ? 'var(--orange)' : 'var(--green)'} height={12} />
        <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>
          {dailyLimitTracker.pctUsed}% of daily limit used | Safe buffer: ${(dailyLimitTracker.remaining - dailyLimitTracker.worstCaseLoss).toLocaleString()}
        </p>
      </Card>

      {/* Worst Case Analysis */}
      <Card>
        <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Worst Case Analysis (Monte Carlo)</h3>
        <ChartPlaceholder label="Monte Carlo drawdown projection: 1000 simulations, 95th/99th percentile lines, current trajectory" height={300} />
        <div className="flex items-center gap-6 mt-3">
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>95th Percentile DD</p>
            <p className="text-sm font-semibold" style={{ color: 'var(--orange)' }}>-4.2%</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>99th Percentile DD</p>
            <p className="text-sm font-semibold" style={{ color: 'var(--red)' }}>-6.8%</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Probability of Ruin</p>
            <p className="text-sm font-semibold" style={{ color: 'var(--green)' }}>0.3%</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Expected Max DD</p>
            <p className="text-sm font-semibold">-3.1%</p>
          </div>
        </div>
      </Card>
    </div>
  );
}

// ---- 5. Account ----
function AccountTab() {
  const [modalDate, setModalDate] = useState<string | null>(null);
  const [activeModalTab, setActiveModalTab] = useState('Trades');
  const modalEntry = mockLedger.find((e) => e.date === modalDate);

  return (
    <div>
      {/* Balance Metrics */}
      <div className="grid grid-cols-4 gap-3 mb-6">
        <MetricCard label="Current Balance" value={`$${accountMetrics.currentBalance.toLocaleString()}`} />
        <MetricCard label="Starting Balance" value={`$${accountMetrics.startingBalance.toLocaleString()}`} />
        <MetricCard label="Net Deposits" value={`$${accountMetrics.netDeposits.toLocaleString()}`} />
        <MetricCard label="Trading P&L" value={`$${accountMetrics.tradingPnL}`} delta={`+$${accountMetrics.tradingPnL}`} positive />
      </div>

      {/* Balance History Chart */}
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Balance History</h3>
        <div className="rounded-lg p-4" style={{ background: 'var(--bg-input)', height: 200 }}>
          <svg width="100%" height="100%" viewBox="0 0 500 160" preserveAspectRatio="none">
            <defs>
              <linearGradient id="balGradV2" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--accent)" stopOpacity={0.3} />
                <stop offset="100%" stopColor="var(--accent)" stopOpacity={0} />
              </linearGradient>
            </defs>
            <path d="M0,150 L50,148 L100,140 L150,120 L200,135 L250,115 L300,100 L350,95 L400,80 L450,70 L500,55" fill="url(#balGradV2)" stroke="none" />
            <path d="M0,150 L50,148 L100,140 L150,120 L200,135 L250,115 L300,100 L350,95 L400,80 L450,70 L500,55" fill="none" stroke="var(--accent)" strokeWidth="2" />
            {/* Deposit marker */}
            <circle cx="150" cy="120" r="4" fill="var(--green)" />
            <text x="155" y="115" fill="var(--text-muted)" fontSize="8" fontFamily="inherit">+$5,000</text>
          </svg>
        </div>
      </Card>

      {/* Ledger */}
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Ledger</h3>

        {/* Header */}
        <div className="grid grid-cols-12 gap-2 pb-2 border-b" style={{ borderColor: 'var(--border)' }}>
          <p className="col-span-2 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Date</p>
          <p className="col-span-3 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Type</p>
          <p className="col-span-2 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Amount</p>
          <p className="col-span-3 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Summary</p>
          <p className="col-span-2 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Action</p>
        </div>

        {/* Rows */}
        {mockLedger.map((entry) => (
          <div
            key={entry.date}
            className="grid grid-cols-12 gap-2 py-3 border-b items-center"
            style={{ borderColor: 'var(--border)' }}
          >
            <p className="col-span-2 text-sm">{entry.date}</p>
            <p className="col-span-3 text-sm" style={{ color: 'var(--text-secondary)' }}>{entry.type}</p>
            <p className="col-span-2 text-sm font-medium" style={{ color: entry.amount >= 0 ? 'var(--green)' : 'var(--red)' }}>
              {entry.amount >= 0 ? '+' : ''}${entry.amount.toFixed(2)}
            </p>
            <p className="col-span-3 text-xs" style={{ color: 'var(--text-muted)' }}>{entry.summary}</p>
            <div className="col-span-2">
              {entry.type.includes('Trading') && (
                <button
                  onClick={() => { setModalDate(entry.date); setActiveModalTab('Trades'); }}
                  className="px-3 py-1 rounded text-xs"
                  style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
                >
                  Details
                </button>
              )}
            </div>
          </div>
        ))}
      </Card>

      {/* Journal Section */}
      <Card>
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Trading Journal</h3>
        <div className="space-y-3">
          {mockLedger.filter((e) => e.journal).map((entry) => (
            <div key={entry.date} className="rounded-lg p-4 border" style={{ borderColor: 'var(--border)', background: 'var(--bg-input)' }}>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-3">
                  <p className="text-sm font-medium">{entry.date}</p>
                  <span
                    className="text-xs px-2 py-0.5 rounded"
                    style={{
                      background: entry.journal!.mood === 'Frustrated' ? 'var(--red-muted)' : 'var(--green-muted)',
                      color: entry.journal!.mood === 'Frustrated' ? 'var(--red)' : 'var(--green)',
                    }}
                  >
                    {entry.journal!.mood}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Confidence:</span>
                  <ConfidenceStars rating={entry.journal!.confidence} />
                </div>
              </div>
              <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>{entry.journal!.notes}</p>
              <div className="mt-2 flex items-center gap-2">
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Day P&L:</span>
                <span className="text-xs font-medium" style={{ color: entry.amount >= 0 ? 'var(--green)' : 'var(--red)' }}>
                  {entry.amount >= 0 ? '+' : ''}${entry.amount.toFixed(2)}
                </span>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Daily Detail Modal */}
      <Modal
        title={`Daily Detail \u2014 ${modalDate || ''}`}
        isOpen={!!modalDate}
        onClose={() => setModalDate(null)}
        width="700px"
      >
        {/* Modal tabs */}
        <div className="flex gap-1 border-b mb-5" style={{ borderColor: 'var(--border)' }}>
          {['Trades', 'Portfolio Changes', 'Notes'].map((t) => (
            <button
              key={t}
              onClick={() => setActiveModalTab(t)}
              className="px-4 py-2 text-sm"
              style={{
                color: activeModalTab === t ? 'var(--accent)' : 'var(--text-muted)',
                borderBottom: activeModalTab === t ? '2px solid var(--accent)' : '2px solid transparent',
                marginBottom: '-1px',
              }}
            >
              {t}
            </button>
          ))}
        </div>

        {activeModalTab === 'Trades' && modalEntry && (
          <div>
            <div className="rounded-lg p-4 mb-4" style={{ background: 'var(--accent-muted)' }}>
              <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>Day Total P&L</p>
              <p className="text-2xl font-bold" style={{ color: modalEntry.amount >= 0 ? 'var(--green)' : 'var(--red)' }}>
                {modalEntry.amount >= 0 ? '+' : ''}${modalEntry.amount.toFixed(2)}
              </p>
            </div>

            {/* Trade rows */}
            <div className="grid grid-cols-12 gap-2 pb-2 border-b" style={{ borderColor: 'var(--border)' }}>
              <p className="col-span-1 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>#</p>
              <p className="col-span-8 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Trade</p>
              <p className="col-span-3 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>P&L</p>
            </div>
            {modalEntry.trades.map((trade, i) => (
              <div key={i} className="grid grid-cols-12 gap-2 py-2.5 border-b" style={{ borderColor: 'var(--border)' }}>
                <p className="col-span-1 text-sm" style={{ color: 'var(--text-muted)' }}>{i + 1}</p>
                <p className="col-span-8 text-sm" style={{ color: 'var(--text-secondary)' }}>{trade.note}</p>
                <p className="col-span-3 text-sm font-medium" style={{ color: trade.amount >= 0 ? 'var(--green)' : 'var(--red)' }}>
                  {trade.amount >= 0 ? '+' : ''}${trade.amount.toFixed(2)}
                </p>
              </div>
            ))}
          </div>
        )}

        {activeModalTab === 'Portfolio Changes' && modalEntry && (
          <div>
            {modalEntry.changes.length > 0 ? (
              modalEntry.changes.map((change, i) => (
                <div key={i} className="flex items-center gap-3 py-3 border-b" style={{ borderColor: 'var(--border)' }}>
                  <span className="text-xs px-2 py-0.5 rounded" style={{ background: 'var(--orange-muted)', color: 'var(--orange)' }}>
                    Change
                  </span>
                  <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>{change}</span>
                </div>
              ))
            ) : (
              <p style={{ color: 'var(--text-muted)' }}>No portfolio changes on this date.</p>
            )}
          </div>
        )}

        {activeModalTab === 'Notes' && (
          <div>
            {modalEntry?.journal ? (
              <div className="mb-4 rounded-lg p-4" style={{ background: 'var(--bg-input)' }}>
                <div className="flex items-center gap-3 mb-2">
                  <span className="text-xs px-2 py-0.5 rounded" style={{
                    background: modalEntry.journal.mood === 'Frustrated' ? 'var(--red-muted)' : 'var(--green-muted)',
                    color: modalEntry.journal.mood === 'Frustrated' ? 'var(--red)' : 'var(--green)',
                  }}>
                    {modalEntry.journal.mood}
                  </span>
                  <ConfidenceStars rating={modalEntry.journal.confidence} />
                </div>
                <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>{modalEntry.journal.notes}</p>
              </div>
            ) : null}
            <textarea
              className="w-full rounded-lg p-3 text-sm"
              style={{
                background: 'var(--bg-input)',
                border: '1px solid var(--border)',
                color: 'var(--text-primary)',
                resize: 'vertical',
              }}
              rows={5}
              placeholder="Add your thoughts, observations, or context for this trading day..."
              defaultValue={modalEntry?.journal?.notes || ''}
            />
            <div className="flex items-center gap-3 mt-3">
              <select
                className="rounded-lg px-3 py-2 text-sm"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                defaultValue={modalEntry?.journal?.mood || ''}
              >
                <option value="">Mood...</option>
                {['Focused', 'Confident', 'Calm', 'Anxious', 'Frustrated', 'Excited'].map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
              <select
                className="rounded-lg px-3 py-2 text-sm"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                defaultValue={modalEntry?.journal?.confidence || 3}
              >
                {[1, 2, 3, 4, 5].map((c) => (
                  <option key={c} value={c}>Confidence: {c}/5</option>
                ))}
              </select>
              <button
                className="px-4 py-2 rounded-lg text-sm font-medium ml-auto"
                style={{ background: 'var(--accent)', color: 'white' }}
              >
                Save Note
              </button>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}

// ---- 6. Webhooks ----
function WebhooksTab() {
  const [testingIdx, setTestingIdx] = useState<number | null>(null);

  return (
    <div>
      {/* Webhook Configs per strategy */}
      <div className="space-y-4 mb-6">
        {webhookConfigs.map((wh, i) => (
          <Card key={i}>
            <div className="flex items-start justify-between mb-3">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <p className="font-medium">{wh.strategy}</p>
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
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Template: {wh.template}</p>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => setTestingIdx(testingIdx === i ? null : i)}
                  className="px-3 py-1.5 rounded-lg text-xs font-medium"
                  style={{
                    background: testingIdx === i ? 'var(--green-muted)' : 'var(--bg-input)',
                    border: '1px solid var(--border)',
                    color: testingIdx === i ? 'var(--green)' : 'var(--text-secondary)',
                  }}
                >
                  {testingIdx === i ? 'Sent!' : 'Test'}
                </button>
                <button
                  className="px-3 py-1.5 rounded-lg text-xs font-medium"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
                >
                  Edit
                </button>
              </div>
            </div>

            {/* Webhook URL */}
            <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)' }}>
              <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Webhook URL</p>
              {wh.url ? (
                <p className="text-sm font-mono truncate" style={{ color: 'var(--text-secondary)' }}>{wh.url}</p>
              ) : (
                <p className="text-sm italic" style={{ color: 'var(--text-muted)' }}>No webhook URL configured</p>
              )}
            </div>
          </Card>
        ))}
      </div>

      {/* Link to webhook templates */}
      <Card className="mb-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Webhook Templates</p>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Manage reusable templates for Discord, Slack, and custom webhooks</p>
          </div>
          <button className="px-3 py-1.5 rounded-lg text-xs font-medium" style={{ background: 'var(--accent)', color: 'white' }}>
            Manage Templates
          </button>
        </div>
      </Card>

      {/* Delivery History */}
      <Card>
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Delivery History</h3>

        <div className="grid grid-cols-7 gap-2 pb-2 border-b" style={{ borderColor: 'var(--border)' }}>
          <p className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Time</p>
          <p className="col-span-2 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Strategy</p>
          <p className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Event</p>
          <p className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Status</p>
          <p className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Response</p>
          <p className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Latency</p>
        </div>

        <div style={{ maxHeight: 400, overflowY: 'auto' }}>
          {webhookDeliveries.map((d, i) => (
            <div key={i} className="grid grid-cols-7 gap-2 py-2.5 border-b items-center" style={{ borderColor: 'var(--border)' }}>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{d.time}</p>
              <p className="col-span-2 text-sm">{d.strategy}</p>
              <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>{d.event}</p>
              <span
                className="text-xs px-1.5 py-0.5 rounded text-center"
                style={{
                  background: d.status === 'success' ? 'var(--green-muted)' : 'var(--red-muted)',
                  color: d.status === 'success' ? 'var(--green)' : 'var(--red)',
                }}
              >
                {d.status}
              </span>
              <p className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
                {d.responseCode || '--'}
              </p>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{d.latency}</p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

// ---- 7. Deploy ----
function DeployTab() {
  return (
    <div>
      {/* Deployment Status */}
      <Card className="mb-6">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold mb-1">Deployment Status</h3>
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              Environment: {deployStatus.environment}
            </p>
          </div>
          <StatusBadge status={deployStatus.workerStatus} />
        </div>

        <div className="grid grid-cols-4 gap-4">
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Worker Status</p>
            <p className="text-sm font-semibold" style={{ color: 'var(--green)' }}>{deployStatus.workerStatus}</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>API Keys</p>
            <p className="text-sm font-semibold" style={{ color: deployStatus.apiKeysConfigured ? 'var(--green)' : 'var(--red)' }}>
              {deployStatus.apiKeysConfigured ? 'Configured' : 'Missing'}
            </p>
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
      </Card>

      {/* Per-Strategy Monitoring Toggles */}
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Strategy Monitoring</h3>

        <div className="space-y-3">
          {monitoringStrategies.map((strat, i) => (
            <div key={i} className="flex items-center justify-between p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
              <div className="flex items-center gap-3">
                {/* Toggle switch */}
                <button
                  className="relative w-10 h-5 rounded-full transition-colors"
                  style={{ background: strat.enabled ? 'var(--accent)' : 'var(--border)' }}
                >
                  <span
                    className="absolute top-0.5 w-4 h-4 rounded-full transition-transform"
                    style={{
                      background: 'white',
                      left: strat.enabled ? '22px' : '2px',
                    }}
                  />
                </button>
                <div>
                  <p className="text-sm font-medium">{strat.name}</p>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    Last signal: {strat.lastSignal}
                  </p>
                </div>
              </div>
              <StatusBadge status={strat.status} />
            </div>
          ))}
        </div>
      </Card>

      {/* Railway Environment */}
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Railway Environment</h3>
        <div className="space-y-2">
          {[
            { key: 'POLYGON_API_KEY', status: 'Set', masked: 'poly_****...Hk3m' },
            { key: 'SUPABASE_URL', status: 'Set', masked: 'https://****...supabase.co' },
            { key: 'SUPABASE_SERVICE_KEY', status: 'Set', masked: 'eyJh****...Q4Nw' },
            { key: 'DATA_FEED', status: 'Set', masked: 'sip' },
          ].map((env) => (
            <div key={env.key} className="flex items-center justify-between py-2 border-b" style={{ borderColor: 'var(--border)' }}>
              <p className="text-sm font-mono" style={{ color: 'var(--text-secondary)' }}>{env.key}</p>
              <div className="flex items-center gap-3">
                <p className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>{env.masked}</p>
                <span className="text-xs px-1.5 py-0.5 rounded" style={{ background: 'var(--green-muted)', color: 'var(--green)' }}>
                  {env.status}
                </span>
              </div>
            </div>
          ))}
        </div>

        <div className="flex gap-2 mt-4">
          <button className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--accent)', color: 'white' }}>
            Deploy
          </button>
          <button className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>
            Redeploy
          </button>
          <button className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--red-muted)', color: 'var(--red)' }}>
            Stop Worker
          </button>
        </div>
      </Card>

      {/* Monitoring Logs */}
      <Card>
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Monitoring Logs</h3>

        <div className="rounded-lg overflow-hidden" style={{ background: '#0d1117', maxHeight: 400, overflowY: 'auto' }}>
          <div className="p-4 font-mono text-xs space-y-1">
            {recentLogs.map((log, i) => {
              const levelColors: Record<string, string> = {
                INFO: '#58a6ff',
                WARN: '#d29922',
                ERROR: '#f85149',
              };
              return (
                <div key={i} className="flex gap-2">
                  <span style={{ color: '#484f58' }}>{log.time}</span>
                  <span style={{ color: levelColors[log.level] || '#8b949e', fontWeight: 600 }}>[{log.level}]</span>
                  <span style={{ color: '#c9d1d9' }}>{log.message}</span>
                </div>
              );
            })}
          </div>
        </div>
      </Card>
    </div>
  );
}

/* =========================================================================
   MAIN V2 COMPONENT
   ========================================================================= */

export default function PortfolioDetailV2() {
  return (
    <div>
      <PageHeader
        title="My Portfolio"
        subtitle="5 strategies | $80,000 starting balance"
        backHref="/portfolios"
        actions={
          <>
            <button className="px-3 py-1.5 rounded-lg text-sm" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>Refresh</button>
            <button className="px-3 py-1.5 rounded-lg text-sm" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>Update Strategies</button>
            <button className="px-3 py-1.5 rounded-lg text-sm" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>Edit</button>
            <button className="px-3 py-1.5 rounded-lg text-sm" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>Clone</button>
          </>
        }
      />

      <TabBar tabs={TABS}>
        {(tab) => (
          <div>
            {tab === 'Live Dashboard' && <LiveDashboardTab />}
            {tab === 'Performance' && <PerformanceTab />}
            {tab === 'Strategies' && <StrategiesTab />}
            {tab === 'Prop Firm Check' && <PropFirmCheckTab />}
            {tab === 'Account' && <AccountTab />}
            {tab === 'Webhooks' && <WebhooksTab />}
            {tab === 'Deploy' && <DeployTab />}
          </div>
        )}
      </TabBar>
    </div>
  );
}
