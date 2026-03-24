'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
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
    status: 'Healthy' as const, healthScore: 92, active: true, fwdSD: 0.3, alertSD: 0.5,
    riskPerTrade: 100, winRate: 58.3, pf: 2.12, trades: 312, pnlContribution: 1680,
    rDistribution: [1.2, -0.8, 2.1, 0.5, -0.3, 1.8, -0.6, 1.5, 2.3, -0.9, 0.7, 1.1, -0.4, 1.9, 0.3, -0.7, 2.5, 1.0, -0.5, 1.6],
  },
  {
    id: '2', name: 'NVDA LONG - Mass #2', symbol: 'NVDA', direction: 'LONG',
    status: 'Warning' as const, healthScore: 68, active: true, fwdSD: 1.2, alertSD: 1.5,
    riskPerTrade: 100, winRate: 54.0, pf: 2.05, trades: 224, pnlContribution: 1380,
    rDistribution: [0.8, -1.2, 1.5, -0.6, 0.3, 2.0, -0.9, 1.1, -0.4, 1.7, 0.5, -0.8, 1.3, -1.0, 2.2, 0.6, -0.3, 1.8, -0.7, 0.9],
  },
  {
    id: '3', name: 'AAPL LONG - Mass #5', symbol: 'AAPL', direction: 'LONG',
    status: 'Healthy' as const, healthScore: 85, active: true, fwdSD: -0.2, alertSD: 0.1,
    riskPerTrade: 75, winRate: 52.8, pf: 1.78, trades: 186, pnlContribution: 770,
    rDistribution: [0.5, -0.6, 1.0, -0.3, 0.8, 1.2, -0.9, 0.4, -0.5, 1.5, 0.7, -0.4, 1.1, 0.3, -0.8, 0.6, 1.3, -0.7, 0.9, -0.2],
  },
  {
    id: '4', name: 'META LONG - Mass #13', symbol: 'META', direction: 'LONG',
    status: 'Critical' as const, healthScore: 38, active: false, fwdSD: 2.4, alertSD: 2.8,
    riskPerTrade: 50, winRate: 45.5, pf: 1.22, trades: 66, pnlContribution: -120,
    rDistribution: [-0.8, 0.5, -1.1, -0.3, 0.9, -0.6, 0.2, -1.0, 0.4, -0.7, 0.3, -0.5, -0.9, 0.6, -0.4, 0.1, -0.8, 0.7, -0.6, -0.3],
  },
  {
    id: '5', name: 'TSLA LONG - Mass #7', symbol: 'TSLA', direction: 'LONG',
    status: 'Healthy' as const, healthScore: 78, active: true, fwdSD: 0.8, alertSD: 0.6,
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
  'Account', 'Webhooks',
];

// ---- 1. Live Dashboard ----
function LiveDashboardTab() {
  const [bpDate, setBpDate] = useState('2026-03-24');
  const [showTradeChart, setShowTradeChart] = useState<number | null>(null);
  const [anomalyTab, setAnomalyTab] = useState('All');
  const [dataMode, setDataMode] = useState<'Planned' | 'Executed'>('Planned');

  const shiftDate = (days: number) => {
    const d = new Date(bpDate);
    d.setDate(d.getDate() + days);
    setBpDate(d.toISOString().split('T')[0]);
  };

  return (
    <div>
      {/* Visibility disclaimer + data mode toggle */}
      <div className="flex items-center justify-between mb-4 px-3 py-2 rounded-lg" style={{ background: 'var(--accent-muted)', border: '1px solid var(--accent)30' }}>
        <span className="text-xs" style={{ color: 'var(--accent)' }}>
          Dashboard metrics reflect <strong>visible strategies</strong> from the Strategies tab.
        </span>
        <div className="flex items-center gap-4 flex-shrink-0">
          <div className="flex items-center gap-2">
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Data:</span>
            {(['Planned', 'Executed'] as const).map((mode) => (
              <button
                key={mode}
                className="text-xs px-2.5 py-1 rounded-full"
                style={{
                  background: dataMode === mode ? 'var(--bg-card)' : 'transparent',
                  color: dataMode === mode ? 'var(--text-primary)' : 'var(--text-muted)',
                  border: dataMode === mode ? '1px solid var(--border)' : '1px solid transparent',
                  cursor: 'pointer', fontWeight: dataMode === mode ? 600 : 400,
                }}
                onClick={() => setDataMode(mode)}
                title={mode === 'Planned' ? 'Planned quantity — assumes unlimited buying power, shows strategy performance' : 'Executed quantity — actual transactions based on available buying power'}
              >
                {mode}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>TQ:</span>
            <select className="text-xs px-2 py-1 rounded-full" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-primary)', cursor: 'pointer' }} defaultValue="None">
              <option value="None">None</option>
              <option value="ttp">Trade The Pool</option>
              <option value="ftmo">FTMO</option>
              <option value="topstep">Topstep</option>
              <option value="custom">My Custom Rules</option>
            </select>
          </div>
        </div>
      </div>

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
        <ChartPlaceholder label="Benchmark equity line with +/- 1SD and 2SD confidence bands overlaid with actual equity curve (green/orange/red based on deviation)" height={400} />
      </Card>

      {/* Open Positions */}
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
          Open Positions
          <span className="ml-2 text-xs px-1.5 py-0.5 rounded" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
            {openPositions.length}
          </span>
        </h3>

        <div style={{ overflowX: 'auto' }}>
          <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                {['Strategy', 'Symbol', 'Dir', 'Entry', 'Current', 'Unrealized', 'Duration', 'Status', ''].map((h) => (
                  <th key={h} className="text-left py-2 px-3 text-xs font-medium" style={{ color: 'var(--text-muted)', background: 'var(--bg-secondary)' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {openPositions.map((pos, i) => (
                <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                  <td className="py-2.5 px-3 font-medium">{pos.strategy}</td>
                  <td className="py-2.5 px-3">{pos.symbol}</td>
                  <td className="py-2.5 px-3">{pos.direction}</td>
                  <td className="py-2.5 px-3" style={{ color: 'var(--text-secondary)' }}>${pos.entryPrice.toFixed(2)}</td>
                  <td className="py-2.5 px-3" style={{ color: 'var(--text-secondary)' }}>${pos.currentPrice.toFixed(2)}</td>
                  <td className="py-2.5 px-3 font-medium" style={{ color: pos.unrealizedPnL >= 0 ? 'var(--green)' : 'var(--red)' }}>
                    {pos.unrealizedPnL >= 0 ? '+' : ''}${pos.unrealizedPnL.toFixed(2)}
                  </td>
                  <td className="py-2.5 px-3 text-xs" style={{ color: 'var(--text-muted)' }}>{pos.duration}</td>
                  <td className="py-2.5 px-3">
                    <span className="text-xs px-1.5 py-0.5 rounded" style={{ background: i === 0 ? 'var(--green-muted)' : 'var(--orange)' + '20', color: i === 0 ? 'var(--green)' : 'var(--orange)' }}>
                      {i === 0 ? 'Matched' : 'Phantom'}
                    </span>
                  </td>
                  <td className="py-2.5 px-3">
                    <button className="text-xs px-2 py-1 rounded" style={{ background: 'var(--red-muted)', color: 'var(--red)', border: 'none', cursor: 'pointer', whiteSpace: 'nowrap' }}>
                      Close Early
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Buying Power Tracker */}
      <Card className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Buying Power Tracker</h3>
          <div className="flex items-center gap-1">
            <button
              className="text-xs px-2 py-1 rounded"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-muted)', cursor: 'pointer' }}
              onClick={() => shiftDate(-1)}
            >
              &larr;
            </button>
            <input
              type="date"
              value={bpDate}
              onChange={(e) => setBpDate(e.target.value)}
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '4px 8px', borderRadius: '6px', fontSize: '0.75rem' }}
            />
            <button
              className="text-xs px-2 py-1 rounded"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-muted)', cursor: 'pointer' }}
              onClick={() => shiftDate(1)}
            >
              &rarr;
            </button>
          </div>
        </div>
        <div className="grid grid-cols-4 gap-4 mb-4">
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Starting Balance</p>
            <p className="text-lg font-semibold">${buyingPower.startingBalance.toLocaleString()}</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Current Available</p>
            <p className="text-lg font-semibold" style={{ color: 'var(--green)' }}>${buyingPower.available.toLocaleString()}</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Currently Allocated</p>
            <p className="text-lg font-semibold" style={{ color: 'var(--orange)' }}>${buyingPower.allocated.toLocaleString()}</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Peak Allocated Today</p>
            <p className="text-lg font-semibold">$24,800</p>
          </div>
        </div>
        <ChartPlaceholder label={`24-hour buying power chart (${bpDate}): Available BP over time, position open/close events marked, allocation spikes visible`} height={180} />
        <div className="flex items-center justify-between mt-2">
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            Utilization: {buyingPower.utilization}% &middot; Max concurrent positions today: 3
          </p>
          {buyingPower.utilization > 80 && (
            <span className="text-xs px-2 py-0.5 rounded" style={{ background: 'var(--red-muted)', color: 'var(--red)' }}>
              High utilization warning
            </span>
          )}
        </div>
      </Card>

      {/* Anomaly Detection */}
      <Card className="mb-6">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
            Anomaly Detection
            <span className="ml-2 text-xs px-1.5 py-0.5 rounded" style={{ background: 'var(--red-muted)', color: 'var(--red)' }}>4</span>
          </h3>
          <div className="flex gap-1">
            {['All', 'Alert Issues', 'Performance'].map((t) => (
              <button
                key={t}
                className="text-xs px-2.5 py-1 rounded-full"
                style={{
                  background: anomalyTab === t ? 'var(--accent-muted)' : 'var(--bg-input)',
                  color: anomalyTab === t ? 'var(--accent)' : 'var(--text-muted)',
                  border: anomalyTab === t ? '1px solid var(--accent)' : '1px solid var(--border)',
                  cursor: 'pointer',
                }}
                onClick={() => setAnomalyTab(t)}
              >
                {t}
              </button>
            ))}
          </div>
        </div>
        <div className="space-y-2">
          {[
            { severity: 'critical' as const, type: 'Phantom Trade', category: 'Alert Issues', message: 'NVDA LONG - Mass #2: Entry alert fired with no matching backtest trade', time: '10:22 AM' },
            { severity: 'critical' as const, type: 'Unclosed Position', category: 'Alert Issues', message: 'SPY LONG - Mass #1: Position open for 4h 12m (expected max hold: 30m)', time: '09:18 AM' },
            { severity: 'warning' as const, type: 'Missed Trade', category: 'Alert Issues', message: 'AAPL LONG - Mass #5: Backtest trade #23 had no corresponding alert (monitor was active)', time: '10:45 AM' },
            { severity: 'warning' as const, type: 'Consecutive Losses', category: 'Performance', message: 'SPY LONG - Mass #1: 3 consecutive losses (unusual for this strategy)', time: '10:32 AM' },
          ]
            .filter((a) => anomalyTab === 'All' || a.category === anomalyTab)
            .map((anomaly, i) => {
              const severityColors = {
                critical: { bg: 'var(--red-muted)', text: 'var(--red)', label: 'CRITICAL' },
                warning: { bg: 'rgba(255,152,0,0.12)', text: 'var(--orange)', label: 'WARNING' },
              };
              const sc = severityColors[anomaly.severity];
              return (
                <div key={i} className="flex items-start gap-3 p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                  <span className="text-xs px-2 py-0.5 rounded font-mono font-medium flex-shrink-0" style={{ background: sc.bg, color: sc.text }}>
                    {sc.label}
                  </span>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-medium mb-0.5" style={{ color: sc.text }}>{anomaly.type}</p>
                    <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>{anomaly.message}</p>
                  </div>
                  <span className="text-xs flex-shrink-0" style={{ color: 'var(--text-muted)' }}>{anomaly.time}</span>
                </div>
              );
            })}
        </div>
      </Card>

      {/* Trade History — matching Streamlit 13 columns */}
      <Card>
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Trade History</h3>

        <div style={{ overflowX: 'auto' }}>
          <table className="w-full text-sm" style={{ borderCollapse: 'collapse', minWidth: 900 }}>
            <thead>
              <tr>
                {['#', 'Strategy', 'Symbol', 'Dir', 'Entry $', 'Exit $', 'Reason', 'P Qty', 'E Qty', 'R', 'P&L', 'Status', ''].map((h) => (
                  <th key={h} className="text-left py-2 px-2 text-xs font-medium" style={{ color: 'var(--text-muted)', background: 'var(--bg-secondary)', whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {liveTradeHistory.map((trade, i) => (
                <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                  <td className="py-2 px-2 text-xs" style={{ color: 'var(--text-muted)' }}>{i + 1}</td>
                  <td className="py-2 px-2 text-xs">{trade.strategy.split(' - ')[0]}</td>
                  <td className="py-2 px-2 text-xs">{trade.symbol}</td>
                  <td className="py-2 px-2 text-xs">{trade.direction.charAt(0)}</td>
                  <td className="py-2 px-2 text-xs">${trade.entryPrice.toFixed(2)}</td>
                  <td className="py-2 px-2 text-xs">{trade.exitPrice ? `$${trade.exitPrice.toFixed(2)}` : '--'}</td>
                  <td className="py-2 px-2 text-xs" style={{ color: 'var(--text-muted)' }}>{trade.exitPrice ? 'Signal' : '--'}</td>
                  <td className="py-2 px-2 text-xs">10</td>
                  <td className="py-2 px-2 text-xs">{trade.exitPrice ? '10' : '--'}</td>
                  <td className="py-2 px-2 text-xs font-medium" style={{ color: trade.pnl === null ? 'var(--text-muted)' : trade.pnl >= 0 ? 'var(--green)' : 'var(--red)' }}>
                    {trade.pnl === null ? '--' : `${trade.pnl >= 0 ? '+' : ''}${(trade.pnl / 50).toFixed(2)}R`}
                  </td>
                  <td className="py-2 px-2 text-xs font-medium" style={{ color: trade.pnl === null ? 'var(--text-muted)' : trade.pnl >= 0 ? 'var(--green)' : 'var(--red)' }}>
                    {trade.pnl === null ? '--' : `${trade.pnl >= 0 ? '+' : ''}$${trade.pnl.toFixed(2)}`}
                  </td>
                  <td className="py-2 px-2">
                    <span className="text-xs px-1.5 py-0.5 rounded" style={{
                      background: trade.status === 'Open' ? 'var(--accent-muted)' : 'var(--green-muted)',
                      color: trade.status === 'Open' ? 'var(--accent)' : 'var(--green)',
                    }}>
                      {trade.status === 'Open' ? 'Open' : 'Matched'}
                    </span>
                  </td>
                  <td className="py-2 px-2">
                    {trade.exitPrice && (
                      <button
                        className="text-xs px-2 py-1 rounded"
                        style={{ background: 'var(--accent-muted)', color: 'var(--accent)', border: 'none', cursor: 'pointer' }}
                        onClick={() => setShowTradeChart(showTradeChart === i ? null : i)}
                      >
                        Chart
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Trade chart modal */}
        {showTradeChart !== null && (() => {
          const t = liveTradeHistory[showTradeChart];
          return (
            <Modal
              title={`${t.strategy} — ${t.symbol} ${t.direction}`}
              isOpen={true}
              onClose={() => setShowTradeChart(null)}
              width="800px"
            >
              {/* Trade metrics */}
              <div className="grid grid-cols-4 gap-4 mb-4">
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Entry Price</p>
                  <p className="text-sm font-bold">${t.entryPrice.toFixed(2)}</p>
                </div>
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Exit Price</p>
                  <p className="text-sm font-bold">{t.exitPrice ? `$${t.exitPrice.toFixed(2)}` : '--'}</p>
                </div>
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>R-Multiple</p>
                  <p className="text-sm font-bold" style={{ color: (t.pnl ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>
                    {t.pnl ? `${t.pnl >= 0 ? '+' : ''}${(t.pnl / 50).toFixed(2)}R` : '--'}
                  </p>
                </div>
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>P&L ($)</p>
                  <p className="text-sm font-bold" style={{ color: (t.pnl ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>
                    {t.pnl ? `${t.pnl >= 0 ? '+' : ''}$${t.pnl.toFixed(2)}` : '--'}
                  </p>
                </div>
              </div>
              {/* Detail row */}
              <div className="grid grid-cols-4 gap-4 mb-4">
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Entry Time</p>
                  <p className="text-xs font-mono">{t.time}</p>
                </div>
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Exit Reason</p>
                  <p className="text-xs">Signal</p>
                </div>
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Exec Type</p>
                  <p className="text-xs">[C]</p>
                </div>
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Qty / BP Used</p>
                  <p className="text-xs">10 / $1,423</p>
                </div>
              </div>
              <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>
                Entry Slippage: 0.04R &middot; Exit Slippage: -0.02R
              </p>
              {/* Chart */}
              <ChartPlaceholder label="Focused candlestick chart (~30 bars around trade): entry marker (blue triangle-up), exit marker (green/red triangle-down), EMA overlays, confluence heatmap pane, stop loss line, target line" height={400} />
            </Modal>
          );
        })()}
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
                <span className="w-3 h-0.5 inline-block" style={{ background: s.color, opacity: 0.6, borderTop: '1px dashed ' + s.color }} />
                {s.name.split(' - ')[0]}
              </span>
            ))}
            <span className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--text-muted)' }}>
              <span className="w-3 h-0.5 inline-block" style={{ background: 'var(--text-primary)' }} />
              <strong>Combined</strong>
            </span>
          </div>
        </div>
        <ChartPlaceholder label="Combined equity curve: bold white portfolio line + per-strategy dashed colored lines (SPY blue, NVDA green, AAPL orange, META purple, TSLA cyan) layered underneath at 60% opacity. Zero reference line. X-axis: exit timestamps, Y-axis: cumulative P&L ($)" height={320} />
      </Card>

      {/* Drawdown Analysis */}
      <Card className="mb-6">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Drawdown Analysis</h3>
          <div className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-muted)' }}>
            <span className="flex items-center gap-1">
              <span className="w-3 h-0.5 inline-block" style={{ background: 'var(--red)' }} /> Drawdown
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-0.5 inline-block" style={{ background: 'var(--orange)', borderTop: '1px dashed var(--orange)' }} /> Max DD Limit (FTMO)
            </span>
          </div>
        </div>
        <ChartPlaceholder label="Drawdown chart: red line with area fill to zero. Dashed orange line at -10% (FTMO max DD limit). Zero reference line. Shows how close drawdown came to the requirement set threshold." height={220} />
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-3">
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Max Drawdown</p>
            <p className="text-sm font-semibold" style={{ color: 'var(--red)' }}>-$1,680 (-2.1%)</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Profitable Days</p>
            <p className="text-sm font-semibold">72% (18/25)</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Avg Daily P&L</p>
            <p className="text-sm font-semibold" style={{ color: 'var(--green)' }}>+$142 (std: $85)</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Current DD</p>
            <p className="text-sm font-semibold" style={{ color: 'var(--green)' }}>-0.1% (margin: 9.9%)</p>
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

      {/* ---- Risk Analytics ---- */}
      <h3 className="text-sm font-semibold mt-6 mb-4" style={{ color: 'var(--text-primary)' }}>Risk Analytics</h3>

      {/* Daily Peak Capital Deployed */}
      <Card className="mb-4">
        <h4 className="text-sm font-medium mb-3">Daily Peak Capital Deployed</h4>
        <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
          Shows the maximum buying power used on each day across the date range. The red dashed line marks your account balance — days that approach or exceed it indicate buying power constraints that may have affected executed quantities.
        </p>
        <ChartPlaceholder label="Bar chart by day: peak capital deployed per day (blue bars). Red dashed line at $80,000 (account balance / max buying power). Days exceeding threshold highlighted in red. X-axis: dates, Y-axis: peak capital ($)" height={220} />
        <div className="grid grid-cols-4 gap-4 mt-3">
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Highest Peak Day</p>
            <p className="text-sm font-bold">$24,800 (Mar 18)</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Avg Peak / Day</p>
            <p className="text-sm font-bold">$14,200</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Days Near Limit</p>
            <p className="text-sm font-bold" style={{ color: 'var(--green)' }}>0</p>
          </div>
          <div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Max Concurrent Positions</p>
            <p className="text-sm font-bold">4</p>
          </div>
        </div>
      </Card>

      {/* Daily P&L vs Limits */}
      <Card className="mb-4">
        <h4 className="text-sm font-medium mb-3">Daily P&L vs Limits</h4>
        <ChartPlaceholder label="Bar chart: daily P&L colored by compliance (blue normal, orange pause breach, red max loss breach) + reference lines for Max Daily Loss (red) and Daily Pause (orange)" height={220} />
        <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>
          0 days breaching daily pause limit &middot; 0 days breaching max daily loss limit
        </p>
      </Card>

      {/* Worst-Case Analysis */}
      <Card className="mb-4">
        <h4 className="text-sm font-medium mb-3">Worst-Case Analysis</h4>
        <div className="grid grid-cols-3 sm:grid-cols-5 gap-3 mb-4">
          {[
            { label: 'Worst Single Day', value: '-$185' },
            { label: 'Worst Losing Streak', value: '3 days (-$312)' },
            { label: 'Worst 5-Day Rolling DD', value: '-$428' },
            { label: 'Days Breaching Pause', value: '0' },
            { label: 'Days Breaching Max Loss', value: '0' },
          ].map((m) => (
            <div key={m.label}>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{m.label}</p>
              <p className="text-sm font-bold" style={m.value.startsWith('-') ? { color: 'var(--red)' } : undefined}>{m.value}</p>
            </div>
          ))}
        </div>
        <h5 className="text-xs font-medium mb-2" style={{ color: 'var(--text-muted)' }}>Top 5 Worst Days</h5>
        <div style={{ overflowX: 'auto' }}>
          <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                {['Date', 'P&L ($)', 'P&L (%)', 'Cumulative DD %', 'Status'].map((h) => (
                  <th key={h} className="text-left py-2 px-3 text-xs font-medium" style={{ color: 'var(--text-muted)', background: 'var(--bg-secondary)' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                { date: '2026-03-17', pnl: -185, pnlPct: -0.23, cumDD: -2.1, status: 'Normal' },
                { date: '2026-03-10', pnl: -142, pnlPct: -0.18, cumDD: -1.5, status: 'Normal' },
                { date: '2026-03-05', pnl: -118, pnlPct: -0.15, cumDD: -0.8, status: 'Normal' },
                { date: '2026-02-28', pnl: -95, pnlPct: -0.12, cumDD: -0.6, status: 'Normal' },
                { date: '2026-02-21', pnl: -88, pnlPct: -0.11, cumDD: -0.4, status: 'Normal' },
              ].map((row) => (
                <tr key={row.date} style={{ borderBottom: '1px solid var(--border)' }}>
                  <td className="py-2 px-3 text-xs font-mono">{row.date}</td>
                  <td className="py-2 px-3 text-xs" style={{ color: 'var(--red)' }}>${row.pnl}</td>
                  <td className="py-2 px-3 text-xs" style={{ color: 'var(--red)' }}>{row.pnlPct.toFixed(2)}%</td>
                  <td className="py-2 px-3 text-xs">{row.cumDD.toFixed(1)}%</td>
                  <td className="py-2 px-3"><span className="text-xs px-1.5 py-0.5 rounded" style={{ background: 'var(--green-muted)', color: 'var(--green)' }}>{row.status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Monte Carlo Simulation */}
      <Card>
        <h4 className="text-sm font-medium mb-3">Monte Carlo Simulation</h4>
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
            { label: 'Bust Probability', value: '2.4%' },
            { label: 'Daily Pause Prob', value: '8.1%' },
            { label: 'Max Loss Prob', value: '1.2%' },
            { label: 'Median Max DD', value: '-3.8%' },
            { label: '95th Pctl DD', value: '-6.2%' },
            { label: 'Expected Worst Day', value: '-$285' },
          ].map((m) => (
            <div key={m.label}>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{m.label}</p>
              <p className="text-sm font-bold">{m.value}</p>
            </div>
          ))}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <ChartPlaceholder label="Max Drawdown Distribution histogram (50 bins)" height={200} />
          <ChartPlaceholder label="Equity Curve Confidence Bands (5th, 25th, 50th, 75th, 95th percentiles)" height={200} />
        </div>
      </Card>
    </div>
  );
}

// ---- 3. Strategies ----
function StrategiesTab() {
  const [visibility, setVisibility] = useState<Record<string, boolean>>(
    Object.fromEntries(portfolioStrategies.map((s) => [s.id, true]))
  );
  const [activeState, setActiveState] = useState<Record<string, boolean>>(
    Object.fromEntries(portfolioStrategies.map((s) => [s.id, s.active]))
  );
  const [eqXAxis, setEqXAxis] = useState<'time' | 'trade'>('time');
  const [dataView, setDataView] = useState('All Data');
  const [customStart, setCustomStart] = useState('');
  const [customEnd, setCustomEnd] = useState('');

  const toggleVisibility = (id: string) => setVisibility((prev) => ({ ...prev, [id]: !prev[id] }));
  const toggleActive = (id: string) => setActiveState((prev) => ({ ...prev, [id]: !prev[id] }));

  const visibleCount = Object.values(visibility).filter(Boolean).length;
  const activeCount = Object.values(activeState).filter(Boolean).length;

  const EQ_FWD = '#FF9800';
  const EQ_LIVE = '#4CAF50';
  const fmtSD = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}\u03c3`;

  const selectStyle: React.CSSProperties = {
    background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)',
    padding: '4px 8px', borderRadius: '6px', fontSize: '0.75rem',
  };

  return (
    <div>
      {/* Controls row */}
      <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          {visibleCount} visible on dashboard &middot; {activeCount} active (webhooks firing)
        </p>
        <div className="flex items-center gap-3 text-[10px]" style={{ color: 'var(--text-muted)' }}>
          <div className="flex items-center gap-1.5">
            <span>Data:</span>
            <select style={selectStyle} value={dataView} onChange={(e) => setDataView(e.target.value)}>
              {['All Data', 'Last 7 Days', 'Last 30 Days', 'Last 90 Days', 'Backtest Only', 'Forward Only', 'Custom'].map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
            {dataView === 'Custom' && (
              <>
                <input type="date" value={customStart} onChange={(e) => setCustomStart(e.target.value)} style={selectStyle} />
                <span>to</span>
                <input type="date" value={customEnd} onChange={(e) => setCustomEnd(e.target.value)} style={selectStyle} />
              </>
            )}
          </div>
          <div className="w-px h-4" style={{ background: 'var(--border)' }} />
          <div className="flex items-center gap-1.5">
            <span>X-axis:</span>
            {(['time', 'trade'] as const).map((mode) => (
              <button key={mode} onClick={() => setEqXAxis(mode)} className="px-2 py-1 rounded font-medium"
                style={{ background: eqXAxis === mode ? 'var(--accent-muted)' : 'var(--bg-input)', color: eqXAxis === mode ? 'var(--accent)' : 'var(--text-muted)', border: eqXAxis === mode ? '1px solid var(--accent)' : '1px solid var(--border)', cursor: 'pointer' }}>
                {mode === 'time' ? 'Time' : 'Trade #'}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="space-y-4">
        {portfolioStrategies.map((strat) => {
          const isVisible = visibility[strat.id] ?? true;
          const isActive = activeState[strat.id] ?? strat.active;
          return (
            <Card key={strat.id} style={!isVisible ? { opacity: 0.6 } : undefined}>
              {/* Header row */}
              <div className="flex items-start justify-between mb-3">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <p className="font-medium">{strat.name}</p>
                    <StatusBadge status={strat.status} />
                    {!isActive && (
                      <span className="text-xs px-1.5 py-0.5 rounded" style={{ background: 'var(--orange)' + '20', color: 'var(--orange)' }}>Paused</span>
                    )}
                    {!isVisible && (
                      <span className="text-xs px-1.5 py-0.5 rounded" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>Hidden</span>
                    )}
                    <span className="flex-1" />
                  </div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    {strat.symbol} | {strat.direction} | ${strat.riskPerTrade}/trade
                  </p>
                </div>
                {/* Sigma badges */}
                <div className="flex items-center gap-1 flex-shrink-0 ml-4">
                  <span className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded" style={{ color: EQ_FWD, background: EQ_FWD + '18' }}>
                    {fmtSD(strat.fwdSD)}
                  </span>
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>|</span>
                  <span className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded" style={{ color: EQ_LIVE, background: EQ_LIVE + '18' }}>
                    {fmtSD(strat.alertSD)}
                  </span>
                </div>
              </div>

              {/* KPIs */}
              <div className="grid grid-cols-5 gap-4 mb-3">
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Win Rate</p>
                  <p className="text-sm font-semibold">{strat.winRate}%</p>
                </div>
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Profit Factor</p>
                  <p className="text-sm font-semibold">{strat.pf.toFixed(2)}</p>
                </div>
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Daily R</p>
                  <p className="text-sm font-semibold">+{(strat.pnlContribution / strat.trades * 0.8).toFixed(2)}</p>
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

              {/* Equity curve */}
              <div className="rounded-lg overflow-hidden mb-2" style={{ background: 'var(--bg-input)' }}>
                <ChartPlaceholder label={`3-segment equity curve (BT blue, FWD orange, Alerts green) — x-axis: ${eqXAxis === 'time' ? 'time' : 'trade #'}`} height={64} />
              </div>

              {/* Action row */}
              <div className="flex items-center gap-2 pt-2" style={{ borderTop: '1px solid var(--border)' }}>
                <button className="px-3 py-1 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                  View Strategy
                </button>
                <button className="px-3 py-1 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                  View Chart
                </button>
                <button
                  className="px-3 py-1 rounded text-xs"
                  style={{
                    background: isVisible ? 'var(--accent-muted)' : 'var(--bg-input)',
                    border: isVisible ? '1px solid var(--accent)' : '1px solid var(--border)',
                    color: isVisible ? 'var(--accent)' : 'var(--text-muted)',
                    cursor: 'pointer',
                  }}
                  onClick={() => toggleVisibility(strat.id)}
                >
                  {isVisible ? 'Visible' : 'Hidden'}
                </button>
                <button
                  className="px-3 py-1 rounded text-xs"
                  style={{ background: 'var(--red-muted)', color: 'var(--red)', border: 'none', cursor: 'pointer' }}
                >
                  Delete
                </button>
                <span className="flex-1" />
                {/* Active/Paused toggle */}
                <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{isActive ? 'Active' : 'Paused'}</span>
                <div
                  className="relative w-7 h-4 rounded-full cursor-pointer flex-shrink-0"
                  style={{ background: isActive ? 'var(--green)' : 'var(--bg-input)', border: isActive ? 'none' : '1px solid var(--border)' }}
                  onClick={() => toggleActive(strat.id)}
                  title={isActive ? 'Active — webhooks will fire for this strategy' : 'Paused — alerts still track but webhooks are suppressed'}
                >
                  <div className="absolute top-0.5 w-3 h-3 rounded-full transition-all" style={{ background: 'white', left: isActive ? '12px' : '2px' }} />
                </div>
              </div>
            </Card>
          );
        })}
      </div>
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

      {/* Deposit / Withdrawal */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <Card>
          <h4 className="text-sm font-medium mb-3">Add Deposit</h4>
          <div className="flex flex-col gap-3">
            <div>
              <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>Amount ($)</label>
              <input type="number" placeholder="0.00" min="0.01" step="0.01"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '8px 12px', borderRadius: '8px', fontSize: '0.875rem', width: '100%' }} />
            </div>
            <div>
              <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>Date</label>
              <input type="date" defaultValue="2026-03-24"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '8px 12px', borderRadius: '8px', fontSize: '0.875rem', width: '100%' }} />
            </div>
            <div>
              <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>Note (optional)</label>
              <input type="text" placeholder="e.g. Initial funding"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '8px 12px', borderRadius: '8px', fontSize: '0.875rem', width: '100%' }} />
            </div>
            <button className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--green)', color: 'white', border: 'none', cursor: 'pointer' }}>
              Add Deposit
            </button>
          </div>
        </Card>
        <Card>
          <h4 className="text-sm font-medium mb-3">Add Withdrawal</h4>
          <div className="flex flex-col gap-3">
            <div>
              <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>Amount ($)</label>
              <input type="number" placeholder="0.00" min="0.01" step="0.01"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '8px 12px', borderRadius: '8px', fontSize: '0.875rem', width: '100%' }} />
            </div>
            <div>
              <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>Date</label>
              <input type="date" defaultValue="2026-03-24"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '8px 12px', borderRadius: '8px', fontSize: '0.875rem', width: '100%' }} />
            </div>
            <div>
              <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>Note (optional)</label>
              <input type="text" placeholder="e.g. Profit withdrawal"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '8px 12px', borderRadius: '8px', fontSize: '0.875rem', width: '100%' }} />
            </div>
            <button className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--red)', color: 'white', border: 'none', cursor: 'pointer' }}>
              Add Withdrawal
            </button>
          </div>
        </Card>
      </div>

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

      {/* Change History */}
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Change History</h3>
        <div className="space-y-2">
          {[
            { type: 'Strategy Added', desc: 'Added TSLA LONG - Mass #7 to portfolio', time: '2026-03-14 09:15:00', color: 'var(--blue)' },
            { type: 'Risk Adjusted', desc: 'SPY LONG Mass #1 $100 -> $115/trade', time: '2026-03-20 08:30:00', color: 'var(--orange)' },
            { type: 'Requirement Changed', desc: 'Switched from SIM2 to FTMO Challenge - $100K', time: '2026-03-12 14:00:00', color: 'var(--green)' },
            { type: 'Strategy Removed', desc: 'Removed QQQ LONG - Mass #4', time: '2026-03-10 11:20:00', color: 'var(--red)' },
            { type: 'Created', desc: 'Portfolio created with 4 strategies', time: '2026-02-15 14:30:00', color: '#AB47BC' },
          ].map((entry, i) => (
            <div key={i} className="flex items-center gap-3 py-2 border-b" style={{ borderColor: 'var(--border)' }}>
              <span className="text-xs font-medium px-2 py-0.5 rounded-full flex-shrink-0" style={{ color: entry.color, background: entry.color + '20' }}>
                {entry.type}
              </span>
              <span className="text-xs flex-1" style={{ color: 'var(--text-secondary)' }}>{entry.desc}</span>
              <span className="text-xs font-mono flex-shrink-0" style={{ color: 'var(--text-muted)' }}>{entry.time}</span>
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
  return (
    <div>
      {/* Webhook Template Selector */}
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-3">Webhook Template</h3>
        <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
          Select an account-based template that defines how this portfolio communicates with your exchange. Each template contains payloads for all 11 webhook event types.
        </p>
        <div className="flex items-center gap-3">
          <select
            className="flex-1"
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '8px 14px', borderRadius: '8px', fontSize: '0.875rem' }}
            defaultValue="tpl_1"
          >
            <option value="">No template (webhooks disabled)</option>
            <option value="tpl_1">SignalStack — Main Account</option>
            <option value="tpl_2">SignalStack — Paper Account</option>
            <option value="tpl_3">Discord — #trading-alerts</option>
          </select>
          <Link
            href="/alerts/webhook-templates/tpl_1"
            className="px-3 py-2 rounded-lg text-xs font-medium"
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--accent)', textDecoration: 'none', whiteSpace: 'nowrap' }}
          >
            View Template
          </Link>
        </div>
        <div className="flex items-center gap-4 mt-3">
          <label className="flex items-center gap-2 text-xs cursor-pointer" style={{ color: 'var(--text-muted)' }}>
            <input type="checkbox" defaultChecked style={{ accentColor: 'var(--accent)' }} />
            Compliance Breach Alerts
          </label>
        </div>
      </Card>

      {/* Delivery History */}
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-3">Webhook Delivery History</h3>
        <div className="grid grid-cols-7 gap-2 pb-2 border-b" style={{ borderColor: 'var(--border)' }}>
          {['Time', 'Strategy', '', 'Event', 'Status', 'Response', 'Latency'].map((h, i) => (
            <p key={i} className={`text-xs font-medium ${i === 1 ? 'col-span-2' : ''}`} style={{ color: 'var(--text-muted)' }}>{h}</p>
          ))}
        </div>
        <div style={{ maxHeight: 300, overflowY: 'auto' }}>
          {webhookDeliveries.map((d, i) => (
            <div key={i} className="grid grid-cols-7 gap-2 py-2.5 border-b items-center" style={{ borderColor: 'var(--border)' }}>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{d.time}</p>
              <p className="col-span-2 text-sm">{d.strategy}</p>
              <p className="text-xs font-mono" style={{ color: 'var(--text-secondary)' }}>{d.event}</p>
              <span className="text-xs px-1.5 py-0.5 rounded text-center" style={{
                background: d.status === 'success' ? 'var(--green-muted)' : 'var(--red-muted)',
                color: d.status === 'success' ? 'var(--green)' : 'var(--red)',
              }}>
                {d.status}
              </span>
              <p className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>{d.responseCode || '--'}</p>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{d.latency}</p>
            </div>
          ))}
        </div>
      </Card>

    </div>
  );
}

/* =========================================================================
   MAIN V2 COMPONENT
   ========================================================================= */

const PULSE_CSS = `@keyframes pulse { 0%, 100% { transform: scale(1); opacity: 0.5; } 50% { transform: scale(2.2); opacity: 0; } }`;
const EQ_FWD_COLOR = '#FF9800';
const EQ_LIVE_COLOR = '#4CAF50';

const btnSecondary: React.CSSProperties = {
  background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)',
  padding: '6px 14px', borderRadius: '8px', fontSize: '0.875rem', cursor: 'pointer',
};

export default function PortfolioDetailV5() {
  useEffect(() => {
    const id = 'portfolio-detail-pulse-css';
    if (!document.getElementById(id)) {
      const s = document.createElement('style'); s.id = id; s.textContent = PULSE_CSS; document.head.appendChild(s);
    }
  }, []);

  return (
    <div>
      <PageHeader
        title="Main Portfolio"
        backHref="/portfolios"
        actions={
          <>
            <button style={btnSecondary}>Refresh</button>
            <button style={btnSecondary}>Edit</button>
            <button style={btnSecondary}>Clone</button>
            <button style={{ ...btnSecondary, background: 'var(--red-muted)', color: 'var(--red)', border: 'none' }}>Delete</button>
          </>
        }
      />

      {/* Status badges + sigma + pulse dot */}
      <div className="flex items-center gap-3 mb-2 flex-wrap">
        <span className="text-xs font-semibold px-2.5 py-1 rounded-full" style={{ color: 'var(--green)', background: 'var(--green)' + '20' }}>
          On Track
        </span>
        <span className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded" style={{ color: EQ_FWD_COLOR, background: EQ_FWD_COLOR + '18' }}>
          +0.8&sigma;
        </span>
        <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>|</span>
        <span className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded" style={{ color: EQ_LIVE_COLOR, background: EQ_LIVE_COLOR + '18' }}>
          +0.6&sigma;
        </span>
        <span className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--green)' }}>
          <span style={{ position: 'relative', display: 'inline-block', width: 8, height: 8 }}>
            <span style={{ position: 'absolute', inset: 0, borderRadius: '50%', background: 'var(--green)', opacity: 0.5, animation: 'pulse 2s ease-in-out infinite' }} />
            <span style={{ position: 'absolute', inset: '25%', borderRadius: '50%', background: 'var(--green)' }} />
          </span>
          Enabled
        </span>
        {['Live', 'Prop Firm'].map((tag) => (
          <span key={tag} className="text-xs px-2 py-0.5 rounded-full" style={{ color: 'var(--text-muted)', background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
            {tag}
          </span>
        ))}
      </div>

      {/* Meta line */}
      <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
        5 strategies &middot; $80,000 balance &middot; 2% scaling &middot; $250 avg risk &middot; 3.5 trades/day
        <span style={{ color: 'var(--accent)' }}> &middot; SignalStack — Main Account</span>
        <span style={{ color: 'var(--green)' }}> &middot; FTMO Challenge - $100K</span>
      </p>

      <TabBar tabs={TABS}>
        {(tab) => (
          <div>
            {tab === 'Live Dashboard' && <LiveDashboardTab />}
            {tab === 'Performance' && <PerformanceTab />}
            {tab === 'Strategies' && <StrategiesTab />}
            {tab === 'Prop Firm Check' && <PropFirmCheckTab />}
            {tab === 'Account' && <AccountTab />}
            {tab === 'Webhooks' && <WebhooksTab />}
          </div>
        )}
      </TabBar>
    </div>
  );
}
