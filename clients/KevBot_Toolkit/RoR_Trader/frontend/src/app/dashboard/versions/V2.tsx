'use client';

import { useState } from 'react';
import Link from 'next/link';
import Card from '@/components/Card';
import MetricCard from '@/components/MetricCard';

/* ========================================================================
   Mock Data
   ======================================================================== */

const mockKPIs = {
  totalStrategies: 12,
  activeMonitors: 4,
  todayAlerts: 7,
  portfolioPnL: '+$247',
  portfolioPnLDelta: '+1.2%',
  bestStrategy: 'SPY LONG - Mass #1',
  bestStrategyDailyR: '+2.41R',
  activePositions: 2,
};

const mockBestStrategy = {
  id: '1',
  name: 'SPY LONG - Mass #1',
  symbol: 'SPY',
  direction: 'LONG',
  winRate: 58.3,
  profitFactor: 1.89,
  dailyR: 2.41,
  totalTrades: 312,
  maxRDrawdown: -1.8,
  // Equity curve points (cumulative R)
  equityCurve: [0, 0.5, -0.2, 0.8, 1.5, 1.2, 2.0, 2.8, 2.5, 3.2, 4.0, 3.6, 4.5, 5.2, 5.0, 5.8, 6.5, 7.0, 6.4, 7.2, 8.0, 8.5, 9.0, 8.6, 9.5, 10.2, 11.0, 11.5, 12.1],
};

const mockBestPortfolio = {
  id: '1',
  name: 'My Portfolio',
  strategies: 8,
  startingBalance: 80000,
  totalPnL: 4230,
  maxDD: -2.1,
  winRate: 55.2,
  avgDailyPnL: 142,
};

const mockRecentActivity = [
  { time: '10:32 AM', type: 'alert' as const, message: 'NVDA LONG entry signal fired', link: '/strategies/1' },
  { time: '10:15 AM', type: 'alert' as const, message: 'SPY LONG exit -- +1.2R', link: '/strategies/2' },
  { time: '09:45 AM', type: 'monitor' as const, message: 'Monitor started for 4 strategies', link: '/alerts' },
  { time: '09:30 AM', type: 'system' as const, message: 'Market open -- RTH session started', link: null },
  { time: 'Yesterday', type: 'strategy' as const, message: 'Saved TSLA LONG - Mass #5', link: '/strategies/4' },
  { time: 'Yesterday', type: 'portfolio' as const, message: 'Updated My Portfolio -- added META LONG', link: '/portfolios/1' },
];

const quickActions = [
  { label: 'Build Strategy', href: '/strategy-builder', icon: 'plus', desc: 'Create a new strategy from scratch' },
  { label: 'View Portfolios', href: '/portfolios', icon: 'folder', desc: 'Manage your portfolio allocations' },
  { label: 'Check Alerts', href: '/alerts', icon: 'bell', desc: 'View active alerts and signals' },
  { label: 'Configure Packs', href: '/confluence-packs', icon: 'layers', desc: 'Set up indicator confluence packs' },
];

/* ========================================================================
   Helper: SVG Mini Equity Curve
   ======================================================================== */

function MiniEquityCurve({ data, width = 320, height = 80, gradientId = 'eqGrad' }: {
  data: number[];
  width?: number;
  height?: number;
  gradientId?: string;
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
  const color = finalVal >= 0 ? 'var(--green)' : 'var(--red)';
  const colorRgb = finalVal >= 0 ? '76,175,80' : '244,67,54';

  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={`rgba(${colorRgb}, 0.25)`} />
          <stop offset="100%" stopColor={`rgba(${colorRgb}, 0)`} />
        </linearGradient>
      </defs>
      <path d={fillD} fill={`url(#${gradientId})`} />
      <path d={pathD} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" />
    </svg>
  );
}

/* ========================================================================
   Helper: Activity Icon
   ======================================================================== */

function ActivityIcon({ type }: { type: string }) {
  const iconMap: Record<string, { bg: string; symbol: string }> = {
    alert: { bg: 'var(--accent-muted)', symbol: '!' },
    monitor: { bg: 'var(--green-muted)', symbol: '~' },
    system: { bg: 'rgba(156,163,175,0.15)', symbol: 'S' },
    strategy: { bg: 'var(--accent-muted)', symbol: '+' },
    portfolio: { bg: 'rgba(168,85,247,0.15)', symbol: 'P' },
  };
  const { bg, symbol } = iconMap[type] || iconMap.system;

  return (
    <div
      className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0"
      style={{ background: bg, color: 'var(--text-secondary)' }}
    >
      {symbol}
    </div>
  );
}

/* ========================================================================
   Helper: Quick Action Icon
   ======================================================================== */

function QuickActionIcon({ icon }: { icon: string }) {
  const symbols: Record<string, string> = {
    plus: '+',
    folder: 'P',
    bell: 'A',
    layers: 'C',
  };

  return (
    <div
      className="w-10 h-10 rounded-lg flex items-center justify-center text-lg font-bold mb-3"
      style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
    >
      {symbols[icon] || '?'}
    </div>
  );
}

/* ========================================================================
   Dashboard V2 Component
   ======================================================================== */

export default function DashboardV2() {
  const [hasStrategies] = useState(true); // toggle to false to test empty state

  // ---------- EMPTY STATE ----------
  if (!hasStrategies) {
    return (
      <div>
        <h1 className="text-2xl font-bold mb-2">Welcome to RoR Trader</h1>
        <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>
          Build data-backed trading strategies without writing code.
        </p>

        <Card className="mb-6">
          <h3 className="font-semibold mb-3">How it Works</h3>
          <p className="text-sm mb-4" style={{ color: 'var(--text-secondary)' }}>
            RoR Trader uses an Indicator &rarr; Interpreter &rarr; Trigger pipeline
            to eliminate subjective chart reading and quantify every trading decision.
          </p>
          <div className="grid grid-cols-2 gap-4">
            {[
              { step: '1', title: 'Indicators', desc: 'Calculate values on price data (EMA, VWAP, RSI, etc.)' },
              { step: '2', title: 'Interpreters', desc: 'Classify indicator states into clear conditions' },
              { step: '3', title: 'Triggers', desc: 'Fire entry/exit signals when conditions align' },
              { step: '4', title: 'Confluence', desc: 'Layer multiple conditions for high-probability setups' },
            ].map((item) => (
              <div
                key={item.step}
                className="rounded-lg p-3 border"
                style={{ borderColor: 'var(--border)', background: 'var(--bg-input)' }}
              >
                <div className="flex items-center gap-2 mb-1">
                  <span
                    className="w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold"
                    style={{ background: 'var(--accent)', color: 'white' }}
                  >
                    {item.step}
                  </span>
                  <span className="text-sm font-medium">{item.title}</span>
                </div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{item.desc}</p>
              </div>
            ))}
          </div>

          <Link href="/strategy-builder">
            <button
              className="mt-5 px-5 py-2.5 rounded-lg text-sm font-medium"
              style={{ background: 'var(--accent)', color: 'white' }}
            >
              Create Your First Strategy
            </button>
          </Link>
        </Card>
      </div>
    );
  }

  // ---------- FULL DASHBOARD ----------
  return (
    <div>
      {/* ---- Welcome / Status Header ---- */}
      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold mb-1">Dashboard</h1>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
            Welcome to RoR Trader
          </p>
        </div>
        <div className="flex items-center gap-3">
          {/* Data source status */}
          <span
            className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full"
            style={{ background: 'var(--green-muted)', color: 'var(--green)' }}
          >
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--green)' }} />
            Polygon.io Connected
          </span>
          {/* Market session indicator */}
          <span
            className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full"
            style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
          >
            RTH Session
          </span>
        </div>
      </div>

      {/* ---- Overview KPI Row ---- */}
      <div className="grid grid-cols-6 gap-4 mb-6">
        <MetricCard label="Total Strategies" value={String(mockKPIs.totalStrategies)} />
        <MetricCard label="Active Monitors" value={String(mockKPIs.activeMonitors)} />
        <MetricCard label="Today's Alerts" value={String(mockKPIs.todayAlerts)} />
        <MetricCard
          label="Portfolio P&L"
          value={mockKPIs.portfolioPnL}
          delta={mockKPIs.portfolioPnLDelta}
          positive
        />
        <MetricCard
          label="Best Strategy"
          value={mockKPIs.bestStrategyDailyR}
          delta={mockKPIs.bestStrategy}
        />
        <MetricCard label="Active Positions" value={String(mockKPIs.activePositions)} />
      </div>

      {/* ---- Two-Column: Top Strategy + Right Panel ---- */}
      <div className="grid grid-cols-5 gap-6 mb-6">
        {/* Left: Top Strategy Showcase */}
        <div className="col-span-3">
          <Card>
            <div className="flex items-start justify-between mb-3">
              <div>
                <h3 className="text-sm font-medium mb-1" style={{ color: 'var(--text-muted)' }}>
                  Top Strategy
                </h3>
                <h2 className="text-lg font-bold">{mockBestStrategy.name}</h2>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  {mockBestStrategy.symbol} | {mockBestStrategy.direction}
                </p>
              </div>
              <Link href={`/strategies/${mockBestStrategy.id}`}>
                <button
                  className="px-3 py-1.5 rounded-lg text-xs font-medium"
                  style={{ background: 'var(--accent)', color: 'white' }}
                >
                  View Strategy
                </button>
              </Link>
            </div>

            {/* KPI row */}
            <div className="grid grid-cols-5 gap-3 mb-4">
              {[
                { label: 'Win Rate', value: `${mockBestStrategy.winRate}%` },
                { label: 'Profit Factor', value: mockBestStrategy.profitFactor.toFixed(2) },
                { label: 'Daily R', value: `+${mockBestStrategy.dailyR.toFixed(2)}` },
                { label: 'Trades', value: String(mockBestStrategy.totalTrades) },
                { label: 'Max R DD', value: `${mockBestStrategy.maxRDrawdown.toFixed(1)}R` },
              ].map((kpi) => (
                <div key={kpi.label}>
                  <p className="text-xs mb-0.5" style={{ color: 'var(--text-muted)' }}>{kpi.label}</p>
                  <p className="text-sm font-semibold">{kpi.value}</p>
                </div>
              ))}
            </div>

            {/* Mini equity curve */}
            <div className="rounded-lg p-2" style={{ background: 'var(--bg-input)' }}>
              <MiniEquityCurve data={mockBestStrategy.equityCurve} height={100} gradientId="dashStratEq" />
            </div>
          </Card>

          {/* Top Portfolio (below top strategy) */}
          <Card className="mt-4">
            <div className="flex items-start justify-between mb-3">
              <div>
                <h3 className="text-sm font-medium mb-1" style={{ color: 'var(--text-muted)' }}>
                  Top Portfolio
                </h3>
                <h2 className="text-lg font-bold">{mockBestPortfolio.name}</h2>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  {mockBestPortfolio.strategies} strategies | ${mockBestPortfolio.startingBalance.toLocaleString()} starting balance
                </p>
              </div>
              <Link href={`/portfolios/${mockBestPortfolio.id}`}>
                <button
                  className="px-3 py-1.5 rounded-lg text-xs font-medium"
                  style={{ background: 'var(--accent)', color: 'white' }}
                >
                  View Portfolio
                </button>
              </Link>
            </div>

            <div className="grid grid-cols-4 gap-3">
              {[
                { label: 'Total P&L', value: `$${mockBestPortfolio.totalPnL.toLocaleString()}`, positive: true },
                { label: 'Max DD', value: `${mockBestPortfolio.maxDD}%` },
                { label: 'Win Rate', value: `${mockBestPortfolio.winRate}%` },
                { label: 'Avg Daily P&L', value: `$${mockBestPortfolio.avgDailyPnL}` },
              ].map((kpi) => (
                <div key={kpi.label}>
                  <p className="text-xs mb-0.5" style={{ color: 'var(--text-muted)' }}>{kpi.label}</p>
                  <p
                    className="text-sm font-semibold"
                    style={kpi.positive ? { color: 'var(--green)' } : undefined}
                  >
                    {kpi.value}
                  </p>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* Right: Recent Activity + System Status */}
        <div className="col-span-2 flex flex-col gap-4">
          {/* Recent Activity Feed */}
          <Card className="flex-1">
            <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-muted)' }}>
              Recent Activity
            </h3>
            <div className="space-y-0">
              {mockRecentActivity.map((activity, idx) => (
                <div
                  key={idx}
                  className="flex items-start gap-3 py-2.5 border-b last:border-0"
                  style={{ borderColor: 'var(--border)' }}
                >
                  <ActivityIcon type={activity.type} />
                  <div className="flex-1 min-w-0">
                    {activity.link ? (
                      <Link href={activity.link}>
                        <p className="text-sm hover:underline" style={{ color: 'var(--text-secondary)' }}>
                          {activity.message}
                        </p>
                      </Link>
                    ) : (
                      <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                        {activity.message}
                      </p>
                    )}
                    <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
                      {activity.time}
                    </p>
                  </div>
                </div>
              ))}
            </div>
            <Link href="/alerts">
              <button
                className="mt-3 w-full py-2 rounded-lg text-xs font-medium border"
                style={{ borderColor: 'var(--border)', color: 'var(--text-secondary)' }}
              >
                View All Alerts
              </button>
            </Link>
          </Card>

          {/* Market / System Status */}
          <Card>
            <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-muted)' }}>
              System Status
            </h3>
            <div className="space-y-2.5">
              {/* Data source */}
              <div className="flex items-center justify-between">
                <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>Data Source</span>
                <span className="text-xs px-2 py-0.5 rounded" style={{ background: 'var(--green-muted)', color: 'var(--green)' }}>
                  Polygon.io
                </span>
              </div>
              {/* Engine status */}
              <div className="flex items-center justify-between">
                <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>Ralph Engine</span>
                <span className="text-xs px-2 py-0.5 rounded" style={{ background: 'var(--green-muted)', color: 'var(--green)' }}>
                  Streaming
                </span>
              </div>
              {/* Market session */}
              <div className="flex items-center justify-between">
                <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>Market Session</span>
                <span className="text-xs px-2 py-0.5 rounded" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                  RTH
                </span>
              </div>
              {/* Forward testing */}
              <div className="flex items-center justify-between">
                <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>Forward Testing</span>
                <span className="text-sm font-medium">3 strategies</span>
              </div>
              {/* Last update */}
              <div className="flex items-center justify-between">
                <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>Last Data Update</span>
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>10:32:15 AM</span>
              </div>
            </div>
          </Card>
        </div>
      </div>

      {/* ---- Quick Actions Grid ---- */}
      <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-muted)' }}>
        Quick Actions
      </h3>
      <div className="grid grid-cols-4 gap-4">
        {quickActions.map((action) => (
          <Link key={action.label} href={action.href}>
            <Card className="cursor-pointer hover:border-opacity-50 h-full">
              <QuickActionIcon icon={action.icon} />
              <h4 className="text-sm font-semibold mb-1">{action.label}</h4>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{action.desc}</p>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
}
