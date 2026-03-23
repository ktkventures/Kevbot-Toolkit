'use client';

import { useState, useRef, useEffect } from 'react';
import Link from 'next/link';
import Card from '@/components/Card';
import Modal from '@/components/Modal';

// =============================================================================
// V7 Meta: Detailed Cockpit — V6 with detail modals on every widget
// =============================================================================
// V6 with detail modals on every widget: equity breakdown, P&L analysis,
// position context, health deep-dive, market analysis, goal tracking, full
// activity log.
// =============================================================================

// ---------------------------------------------------------------------------
// CSS Animations
// ---------------------------------------------------------------------------

const ANIMATION_STYLES = `
@keyframes v7-fade-in {
  0% { opacity: 0; transform: translateY(6px); }
  100% { opacity: 1; transform: translateY(0); }
}
@keyframes v7-glow-border {
  0%, 100% { box-shadow: 0 0 4px rgba(0,255,136,0.08); }
  50% { box-shadow: 0 0 14px rgba(0,255,136,0.22); }
}
@keyframes v7-pulse-dot {
  0%, 100% { opacity: 0.4; transform: scale(1); }
  50% { opacity: 1; transform: scale(1.4); }
}
@keyframes v7-progress-fill {
  0% { width: 0%; }
}
@keyframes v7-regime-pulse {
  0%, 100% { opacity: 0.6; }
  50% { opacity: 1; }
}
@keyframes v7-slide-in {
  0% { transform: translateX(12px); opacity: 0; }
  100% { transform: translateX(0); opacity: 1; }
}
`;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface WidgetConfig {
  id: string;
  label: string;
  enabled: boolean;
}

interface KpiConfig {
  id: string;
  label: string;
  value: string;
  delta?: string;
  positive?: boolean;
  enabled: boolean;
}

interface PortfolioOption {
  id: string;
  name: string;
  selected: boolean;
}

// ---------------------------------------------------------------------------
// Mock Data
// ---------------------------------------------------------------------------

const MOCK_PORTFOLIO = {
  todayPnL: 247,
  todayPnLPct: 0.31,
  profitFactor: 1.89,
  openPositions: 2,
  monthlyTarget: 5000,
  monthlyProgress: 3247,
};

const MOCK_MARKET = {
  regime: 'Bull' as const,
  vix: 14.2,
  spy: { price: 576.42, change: +0.87, changePct: +0.15 },
  qqq: { price: 487.15, change: +2.34, changePct: +0.48 },
  breadth: { advancers: 312, decliners: 188, ratio: 1.66 },
};

const MOCK_SYSTEM = {
  polygon: true,
  ralph: true,
  workerUptime: '4d 12h 33m',
  lastUpdate: '10:32:15 AM',
};

const MOCK_POSITIONS = [
  { id: '1', strategy: 'SPY LONG - Mass #1', symbol: 'SPY', direction: 'LONG' as const, entryPrice: 574.80, currentPrice: 576.42, unrealizedR: 1.35, entryTime: '09:35 AM', matched: true },
  { id: '2', strategy: 'NVDA LONG - Momentum', symbol: 'NVDA', direction: 'LONG' as const, entryPrice: 896.50, currentPrice: 901.20, unrealizedR: 0.62, entryTime: '10:12 AM', matched: true },
];

// Health data — deviation_sd per strategy/portfolio from classify_strategy_health()
const MOCK_HEALTH_STRATEGIES = [
  { id: '1', name: 'SPY LONG - Mass #1', deviationSd: 0.42, status: 'on_track' as const, alertTrades: 23, expectedR: 28.5, actualR: 30.2 },
  { id: '2', name: 'NVDA LONG - Momentum', deviationSd: -0.85, status: 'on_track' as const, alertTrades: 18, expectedR: 22.0, actualR: 18.3 },
  { id: '3', name: 'AAPL LONG - Mass #5', deviationSd: 1.72, status: 'outperforming' as const, alertTrades: 31, expectedR: 15.0, actualR: 24.8 },
  { id: '4', name: 'TSLA LONG - Mass #5', deviationSd: -1.95, status: 'underperforming' as const, alertTrades: 15, expectedR: 18.0, actualR: 8.2 },
  { id: '5', name: 'META LONG - Mass #13', deviationSd: 0.11, status: 'on_track' as const, alertTrades: 8, expectedR: 5.0, actualR: 5.3 },
];

const MOCK_HEALTH_PORTFOLIOS = [
  { id: 'p1', name: 'My Portfolio', deviationSd: 0.35, status: 'on_track' as const, alertTrades: 67, expectedPnL: 3200, actualPnL: 3450 },
  { id: 'p2', name: 'Scalping Portfolio', deviationSd: -1.62, status: 'underperforming' as const, alertTrades: 42, expectedPnL: 1800, actualPnL: 920 },
];

const MOCK_ISSUES = [
  { id: '1', severity: 'warning' as const, message: 'NVDA LONG \u2014 position open 3h, exceeds avg duration (45min)', link: '/strategies/2' },
  { id: '2', severity: 'info' as const, message: 'SPY LONG exit price drift: alert $576.42 vs backtest $576.80 (-$0.38)', link: '/strategies/1' },
  { id: '3', severity: 'error' as const, message: 'Webhook delivery failed for TradeThePool template \u2014 3 retries exhausted', link: '/alerts/webhook-templates' },
];

const MOCK_ALERTS = [
  { time: '10:32 AM', type: 'entry' as const, message: 'NVDA LONG entry signal fired [C]' },
  { time: '10:15 AM', type: 'exit' as const, message: 'SPY LONG exit \u2014 +1.2R [C]' },
  { time: '09:45 AM', type: 'system' as const, message: 'Monitor started for 4 strategies' },
  { time: '09:30 AM', type: 'system' as const, message: 'Market open \u2014 RTH session' },
];

const CALENDAR_DATA: Record<number, number> = {
  1: 142, 2: -67, 3: 210, 4: 55, 5: -120,
  8: 180, 9: 95, 10: -42, 11: 310, 12: -85,
  15: 165, 16: 220, 17: -30, 18: 145, 19: 280,
};

// Equity curve data (cumulative P&L over 30 data points)
const EQUITY_DATA = [
  0, 142, 75, 285, 340, 220, 220, 220,
  400, 495, 453, 763, 678, 678, 678,
  843, 1063, 1033, 1178, 1458, 1458, 1458,
  1600, 1750, 1820, 1900, 2050, 2200, 2350, 2500,
];

// Daily P&L for bar chart
const DAILY_PNL = [
  { day: 'Mar 1', value: 142 }, { day: 'Mar 2', value: -67 }, { day: 'Mar 3', value: 210 },
  { day: 'Mar 4', value: 55 }, { day: 'Mar 5', value: -120 }, { day: 'Mar 8', value: 180 },
  { day: 'Mar 9', value: 95 }, { day: 'Mar 10', value: -42 }, { day: 'Mar 11', value: 310 },
  { day: 'Mar 12', value: -85 }, { day: 'Mar 15', value: 165 }, { day: 'Mar 16', value: 220 },
  { day: 'Mar 17', value: -30 }, { day: 'Mar 18', value: 145 }, { day: 'Mar 19', value: 280 },
];

// ---------------------------------------------------------------------------
// V7 Additional Mock Data for Modals
// ---------------------------------------------------------------------------

const MOCK_EQUITY_SEGMENTS = {
  btTotalR: 18.4,
  fwTotalR: 7.2,
  liveTotalR: 4.6,
};

const MOCK_EQUITY_INFLECTIONS = [
  { label: 'Biggest single-day win', value: '+$310', date: 'Mar 11' },
  { label: 'Worst drawdown', value: '-$120', date: 'Mar 5-8' },
  { label: 'Longest win streak', value: '4 days', date: 'Mar 15-19' },
];

const MOCK_PNL_STATS = {
  totalDays: 15,
  winDays: 10,
  lossDays: 5,
  avgWinDay: 195.2,
  avgLossDay: -68.8,
  bestDay: { value: 310, date: 'Mar 11' },
  worstDay: { value: -120, date: 'Mar 5' },
  currentStreak: { type: 'win' as const, days: 3 },
};

const MOCK_PNL_WEEKLY = [
  { week: 'Week 1 (Mar 1-5)', pnl: 220, trades: 12, winRate: 58 },
  { week: 'Week 2 (Mar 8-12)', pnl: 458, trades: 14, winRate: 64 },
  { week: 'Week 3 (Mar 15-19)', pnl: 780, trades: 11, winRate: 73 },
];

const MOCK_MONTHLY_SUMMARY = {
  prevMonthPnL: 1820,
  prevMonthWinRate: 52,
  prevMonthTradingDays: 20,
  consistencyScore: 73,
  consistencyExplanation: 'Consistency score measures the % of trading days that ended green, weighted by how close each day\'s P&L was to your average. A score above 60 indicates reliable execution.',
  goalProgress: { target: 5000, current: 3247, percentComplete: 64.9 },
};

const MOCK_POSITION_DETAILS: Record<string, {
  strategyId: string;
  entryTrigger: string;
  confluenceConditions: string[];
  stopLevel: number;
  targetLevel: number;
  avgDuration: string;
  timeInTrade: string;
  btWinRate: number;
  btAvgR: number;
  recentTrades: { date: string; result: string; r: number }[];
}> = {
  '1': {
    strategyId: 's1',
    entryTrigger: 'EMA Stack Bullish [C]',
    confluenceConditions: ['EMA_STACK: SML', 'MACD_LINE: Bull', 'RVOL: Above Avg'],
    stopLevel: 573.20,
    targetLevel: 580.50,
    avgDuration: '45 min',
    timeInTrade: '57 min',
    btWinRate: 58.3,
    btAvgR: 1.42,
    recentTrades: [
      { date: 'Mar 19', result: 'WIN', r: 1.8 },
      { date: 'Mar 18', result: 'WIN', r: 0.9 },
      { date: 'Mar 16', result: 'LOSS', r: -1.0 },
      { date: 'Mar 15', result: 'WIN', r: 2.1 },
      { date: 'Mar 12', result: 'LOSS', r: -0.8 },
    ],
  },
  '2': {
    strategyId: 's2',
    entryTrigger: 'Momentum Breakout [L0]',
    confluenceConditions: ['MACD_LINE: Bull', 'VWAP: Above', 'ATR: Expanding'],
    stopLevel: 890.00,
    targetLevel: 915.00,
    avgDuration: '1h 20min',
    timeInTrade: '20 min',
    btWinRate: 52.1,
    btAvgR: 1.65,
    recentTrades: [
      { date: 'Mar 19', result: 'WIN', r: 2.3 },
      { date: 'Mar 17', result: 'LOSS', r: -1.0 },
      { date: 'Mar 15', result: 'WIN', r: 1.1 },
      { date: 'Mar 11', result: 'WIN', r: 3.2 },
      { date: 'Mar 10', result: 'LOSS', r: -0.5 },
    ],
  },
};

const MOCK_HEALTH_DETAILS: Record<string, {
  explanation: string;
  recommendation: string;
  healthyLastWeek: boolean;
  correlations: string[];
  expectedRHistory: number[];
  actualRHistory: number[];
}> = {
  '1': {
    explanation: 'SPY LONG - Mass #1 is performing within 0.42 SD of its backtest expectation. This indicates the live environment closely matches backtest conditions.',
    recommendation: 'No action needed. Continue monitoring. Consider increasing position size if this consistency holds for 30+ trades.',
    healthyLastWeek: true,
    correlations: ['Positively correlated with SPY momentum', 'Inverse to VIX spikes'],
    expectedRHistory: [5, 10, 15, 20, 25, 28.5],
    actualRHistory: [5.2, 11, 14, 21, 27, 30.2],
  },
  '2': {
    explanation: 'NVDA LONG - Momentum is slightly below expectation at -0.85 SD. This is within normal variance and not yet actionable.',
    recommendation: 'Monitor for another 10 trades. If deviation exceeds -1.5 SD, review entry timing and market regime alignment.',
    healthyLastWeek: true,
    correlations: ['Strongly correlated with QQQ', 'Performs better in low-VIX environments'],
    expectedRHistory: [4, 8, 12, 16, 20, 22],
    actualRHistory: [3.5, 7, 10, 14, 17, 18.3],
  },
  '3': {
    explanation: 'AAPL LONG - Mass #5 is significantly outperforming at +1.72 SD. Live results are 65% above backtest expectations.',
    recommendation: 'Outperformance may be due to favorable market regime. Consider whether recent conditions (low VIX, strong sector momentum) are sustainable. Do not increase sizing based on recent outperformance alone.',
    healthyLastWeek: true,
    correlations: ['Follows XLK sector momentum', 'Earnings season boost detected'],
    expectedRHistory: [3, 6, 9, 12, 14, 15],
    actualRHistory: [4, 8, 14, 18, 22, 24.8],
  },
  '4': {
    explanation: 'TSLA LONG - Mass #5 is significantly underperforming at -1.95 SD. Live cumulative R is 54% below backtest expectation.',
    recommendation: 'Review strategy parameters. TSLA may have shifted into a regime unfavorable for this strategy type. Consider pausing until underperformance stabilizes or regime changes.',
    healthyLastWeek: false,
    correlations: ['Negatively impacted by recent TSLA earnings miss', 'High beta amplifies drawdowns'],
    expectedRHistory: [3, 6, 10, 14, 16, 18],
    actualRHistory: [2, 4, 5, 6, 7, 8.2],
  },
  '5': {
    explanation: 'META LONG - Mass #13 is performing almost exactly at expectation with +0.11 SD. Only 8 trades so far, so confidence is moderate.',
    recommendation: 'Continue running. Need 20+ trades for high-confidence health assessment.',
    healthyLastWeek: true,
    correlations: ['Follows communication services sector', 'Moderate VIX sensitivity'],
    expectedRHistory: [1, 2, 3, 4, 5],
    actualRHistory: [1.1, 2.2, 3, 4.2, 5.3],
  },
  'p1': {
    explanation: 'My Portfolio is tracking close to expectations at +0.35 SD. The diversification across strategies is working well.',
    recommendation: 'Portfolio health is strong. The underperformance in TSLA is being offset by AAPL outperformance.',
    healthyLastWeek: true,
    correlations: ['Broad market exposure', 'Low single-strategy concentration risk'],
    expectedRHistory: [500, 1000, 1600, 2200, 2800, 3200],
    actualRHistory: [520, 1050, 1700, 2400, 3000, 3450],
  },
  'p2': {
    explanation: 'Scalping Portfolio is underperforming at -1.62 SD. Slippage and execution timing may be contributing factors.',
    recommendation: 'Review execution speed and fill quality. Scalping strategies are more sensitive to latency. Consider switching to limit-order execution types.',
    healthyLastWeek: false,
    correlations: ['Sensitive to spread widening', 'Performs poorly in low-volume environments'],
    expectedRHistory: [300, 600, 900, 1200, 1500, 1800],
    actualRHistory: [250, 450, 580, 700, 820, 920],
  },
};

const MOCK_VIX_HISTORY = [
  { day: 'Feb 21', value: 18.5 }, { day: 'Feb 22', value: 17.8 }, { day: 'Feb 23', value: 16.9 },
  { day: 'Feb 24', value: 17.2 }, { day: 'Feb 25', value: 16.1 },
  { day: 'Feb 28', value: 15.5 }, { day: 'Mar 1', value: 15.8 }, { day: 'Mar 2', value: 16.2 },
  { day: 'Mar 3', value: 15.4 }, { day: 'Mar 4', value: 14.9 },
  { day: 'Mar 5', value: 15.8 }, { day: 'Mar 8', value: 15.2 }, { day: 'Mar 9', value: 14.8 },
  { day: 'Mar 10', value: 15.1 }, { day: 'Mar 11', value: 14.5 },
  { day: 'Mar 12', value: 14.9 }, { day: 'Mar 15', value: 14.3 }, { day: 'Mar 16', value: 14.0 },
  { day: 'Mar 17', value: 14.6 }, { day: 'Mar 18', value: 14.1 },
  { day: 'Mar 19', value: 14.2 },
];

const MOCK_REGIME_HISTORY = [
  { regime: 'Neutral', startDate: 'Feb 1', endDate: 'Feb 14', days: 14 },
  { regime: 'Bear', startDate: 'Feb 15', endDate: 'Feb 28', days: 14 },
  { regime: 'Bull', startDate: 'Mar 1', endDate: 'Mar 15', days: 15 },
  { regime: 'Bull', startDate: 'Mar 15', endDate: 'present', days: 7 },
];

const MOCK_GOAL_WEEKLY_PROGRESS = [
  { week: 'Week 1', earned: 220, target: 1250, pace: 'behind' as const },
  { week: 'Week 2', earned: 458, target: 1250, pace: 'behind' as const },
  { week: 'Week 3', earned: 780, target: 1250, pace: 'behind' as const },
  { week: 'Week 4 (partial)', earned: 1789, target: 1250, pace: 'ahead' as const },
];

const MOCK_GOAL_HISTORY = [
  { month: 'January', target: 4000, achieved: 3200, hit: false },
  { month: 'February', target: 4500, achieved: 4800, hit: true },
  { month: 'March', target: 5000, achieved: 3247, hit: false },
];

const MOCK_FULL_ACTIVITY = [
  { time: '10:32 AM', type: 'entry' as const, message: 'NVDA LONG entry signal fired [C]' },
  { time: '10:15 AM', type: 'exit' as const, message: 'SPY LONG exit \u2014 +1.2R [C]' },
  { time: '09:45 AM', type: 'system' as const, message: 'Monitor started for 4 strategies' },
  { time: '09:30 AM', type: 'system' as const, message: 'Market open \u2014 RTH session' },
  { time: '09:28 AM', type: 'system' as const, message: 'Pre-market scan completed: 12 symbols' },
  { time: '09:15 AM', type: 'entry' as const, message: 'SPY LONG entry signal fired [C]' },
  { time: 'Yesterday 3:55 PM', type: 'exit' as const, message: 'AAPL LONG exit \u2014 +2.1R [C]' },
  { time: 'Yesterday 3:30 PM', type: 'summary' as const, message: 'Daily summary: 3 trades, +2.8R, $280 realized' },
  { time: 'Yesterday 2:45 PM', type: 'exit' as const, message: 'TSLA LONG exit \u2014 -0.5R [C] (stop hit)' },
  { time: 'Yesterday 1:20 PM', type: 'entry' as const, message: 'TSLA LONG entry signal fired [L0]' },
  { time: 'Yesterday 11:00 AM', type: 'entry' as const, message: 'AAPL LONG entry signal fired [C]' },
  { time: 'Yesterday 9:35 AM', type: 'entry' as const, message: 'META LONG entry signal fired [C]' },
  { time: 'Yesterday 9:30 AM', type: 'system' as const, message: 'Market open \u2014 RTH session' },
  { time: '2 days ago', type: 'summary' as const, message: 'Daily summary: 2 trades, +1.5R, $165 realized' },
  { time: '2 days ago', type: 'system' as const, message: 'Webhook template updated: TradeThePool' },
];

const MOCK_RESOLVED_ISSUES = [
  { id: 'r1', severity: 'warning' as const, message: 'SPY LONG position duration exceeded avg \u2014 resolved by normal exit', resolvedAt: 'Yesterday 3:55 PM' },
  { id: 'r2', severity: 'info' as const, message: 'Monitor reconnected after brief Polygon WS disconnect', resolvedAt: 'Mar 18 2:30 PM' },
  { id: 'r3', severity: 'error' as const, message: 'Webhook delivery restored after endpoint fix', resolvedAt: 'Mar 17 10:00 AM' },
];

// ---------------------------------------------------------------------------
// Default KPI config
// ---------------------------------------------------------------------------

const DEFAULT_KPIS: KpiConfig[] = [
  { id: 'today-pnl', label: 'Today P&L', value: '$247', delta: '+0.31%', positive: true, enabled: true },
  { id: 'pf', label: 'Profit Factor', value: '1.89', enabled: true },
  { id: 'open-positions', label: 'Open Positions', value: '2', enabled: true },
  { id: 'market-regime', label: 'Market Regime', value: 'Bull', enabled: true },
  { id: 'win-rate', label: 'Win Rate', value: '55.2%', enabled: false },
  { id: 'last-30d-pnl', label: 'Last 30 Days P&L', value: '$2,847', delta: '+3.6%', positive: true, enabled: false },
  { id: 'total-pnl', label: 'Total P&L', value: '$4,230', enabled: false },
  { id: 'max-dd', label: 'Max Drawdown', value: '-2.1%', enabled: false },
  { id: 'daily-r', label: 'Avg Daily R', value: '+1.42', enabled: false },
  { id: 'active-strategies', label: 'Active Strategies', value: '8', enabled: false },
  { id: 'alerts-today', label: 'Alerts Today', value: '7', enabled: false },
  { id: 'balance', label: 'Balance', value: '$84,230', enabled: false },
];

// ---------------------------------------------------------------------------
// Default portfolio options
// ---------------------------------------------------------------------------

const INITIAL_PORTFOLIOS: PortfolioOption[] = [
  { id: 'p1', name: 'My Portfolio', selected: true },
  { id: 'p2', name: 'Scalping Portfolio', selected: true },
  { id: 'p3', name: 'Testing Portfolio', selected: false },
  { id: 'p4', name: 'Paper Trading', selected: false },
];

// ---------------------------------------------------------------------------
// Default widget config
// ---------------------------------------------------------------------------

const DEFAULT_WIDGETS: WidgetConfig[] = [
  { id: 'equity-curve', label: 'Portfolio Equity Curve', enabled: true },
  { id: 'daily-pnl', label: 'Daily P&L Bar Chart', enabled: true },
  { id: 'monthly-goal', label: 'Monthly Goal Tracker', enabled: true },
  { id: 'positions', label: 'Active Positions', enabled: true },
  { id: 'health', label: 'Portfolio Health', enabled: true },
  { id: 'market-regime', label: 'Market Regime', enabled: true },
  { id: 'calendar', label: 'P&L Calendar Heatmap', enabled: true },
  { id: 'issues', label: 'Issues & Warnings', enabled: true },
  { id: 'activity', label: 'Recent Activity Feed', enabled: true },
  { id: 'quick-actions', label: 'Quick Actions', enabled: true },
];

// ---------------------------------------------------------------------------
// SVG Components
// ---------------------------------------------------------------------------

/** Portfolio Equity Curve — gradient-filled area chart */
function EquityCurve({
  data,
  width = 700,
  height = 260,
}: {
  data: number[];
  width?: number;
  height?: number;
}) {
  const pad = { top: 20, right: 16, bottom: 28, left: 56 };
  const chartW = width - pad.left - pad.right;
  const chartH = height - pad.top - pad.bottom;

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  const points = data.map((v, i) => {
    const x = pad.left + (i / (data.length - 1)) * chartW;
    const y = pad.top + chartH - ((v - min) / range) * chartH;
    return { x, y };
  });

  const lineD = `M${points.map(p => `${p.x},${p.y}`).join(' L')}`;
  const areaD = `${lineD} L${pad.left + chartW},${pad.top + chartH} L${pad.left},${pad.top + chartH} Z`;
  const lastPt = points[points.length - 1];
  const positive = data[data.length - 1] >= data[0];
  const color = positive ? 'var(--green)' : 'var(--red)';

  // Y-axis ticks
  const yTicks = 5;
  const yTickVals = Array.from({ length: yTicks }, (_, i) => min + (range / (yTicks - 1)) * i);

  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet">
      <defs>
        <linearGradient id="v7-equityGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.2" />
          <stop offset="100%" stopColor={color} stopOpacity="0.02" />
        </linearGradient>
      </defs>

      {/* Grid lines */}
      {yTickVals.map((val, i) => {
        const y = pad.top + chartH - ((val - min) / range) * chartH;
        return (
          <g key={i}>
            <line x1={pad.left} y1={y} x2={pad.left + chartW} y2={y} stroke="var(--border)" strokeWidth="0.5" strokeDasharray="4 4" />
            <text x={pad.left - 8} y={y + 3} textAnchor="end" fontSize="9" fill="var(--text-muted)" fontFamily="monospace">
              ${Math.round(val).toLocaleString()}
            </text>
          </g>
        );
      })}

      {/* Area fill */}
      <path d={areaD} fill="url(#v7-equityGrad)" />

      {/* Line */}
      <path d={lineD} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />

      {/* End dot */}
      <circle cx={lastPt.x} cy={lastPt.y} r="4" fill={color} style={{ animation: 'v7-pulse-dot 2s ease-in-out infinite' }} />
      <circle cx={lastPt.x} cy={lastPt.y} r="8" fill={color} opacity="0.15" />

      {/* Current value label */}
      <text x={lastPt.x} y={lastPt.y - 12} textAnchor="middle" fontSize="10" fill={color} fontWeight="600" fontFamily="monospace">
        ${data[data.length - 1].toLocaleString()}
      </text>
    </svg>
  );
}

/** Daily P&L Bar Chart */
function DailyPnLChart({
  data,
  width = 700,
  height = 220,
}: {
  data: { day: string; value: number }[];
  width?: number;
  height?: number;
}) {
  const pad = { top: 16, right: 16, bottom: 36, left: 50 };
  const chartW = width - pad.left - pad.right;
  const chartH = height - pad.top - pad.bottom;

  const maxAbs = Math.max(...data.map(d => Math.abs(d.value)), 1);
  const barW = Math.max((chartW / data.length) - 4, 8);
  const zeroY = pad.top + chartH / 2;

  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet">
      {/* Zero line */}
      <line x1={pad.left} y1={zeroY} x2={pad.left + chartW} y2={zeroY} stroke="var(--border)" strokeWidth="1" />

      {/* Y-axis labels */}
      <text x={pad.left - 8} y={pad.top + 4} textAnchor="end" fontSize="9" fill="var(--text-muted)" fontFamily="monospace">
        +${maxAbs}
      </text>
      <text x={pad.left - 8} y={zeroY + 3} textAnchor="end" fontSize="9" fill="var(--text-muted)" fontFamily="monospace">
        $0
      </text>
      <text x={pad.left - 8} y={pad.top + chartH + 4} textAnchor="end" fontSize="9" fill="var(--text-muted)" fontFamily="monospace">
        -${maxAbs}
      </text>

      {/* Bars */}
      {data.map((d, i) => {
        const x = pad.left + (i / data.length) * chartW + (chartW / data.length - barW) / 2;
        const barH = (Math.abs(d.value) / maxAbs) * (chartH / 2);
        const y = d.value >= 0 ? zeroY - barH : zeroY;
        const color = d.value >= 0 ? 'var(--green)' : 'var(--red)';

        return (
          <g key={i}>
            <rect x={x} y={y} width={barW} height={barH} rx={2} fill={color} opacity="0.8">
              <title>{d.day}: {d.value >= 0 ? '+' : ''}${d.value}</title>
            </rect>
            {/* Day label (show every other to avoid clutter) */}
            {i % 2 === 0 && (
              <text
                x={x + barW / 2}
                y={pad.top + chartH + 14}
                textAnchor="middle"
                fontSize="8"
                fill="var(--text-muted)"
              >
                {d.day.replace('Mar ', '')}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

/** Market Regime Gauge (compact) */
function RegimeGauge({
  regime,
  size = 90,
}: {
  regime: 'Bull' | 'Bear' | 'Neutral';
  size?: number;
}) {
  const c = size / 2;
  const r = (size - 14) / 2;
  const circumference = 2 * Math.PI * r;
  const arcLen = circumference * 0.75;

  const regimeData = {
    Bull: { pct: 0.78, color: 'var(--green)', label: 'BULL' },
    Bear: { pct: 0.25, color: 'var(--red)', label: 'BEAR' },
    Neutral: { pct: 0.50, color: 'var(--orange)', label: 'NEUTRAL' },
  };

  const { pct, color, label } = regimeData[regime];
  const fillLen = arcLen * pct;

  return (
    <div className="relative flex flex-col items-center">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle
          cx={c} cy={c} r={r} fill="none"
          stroke="var(--bg-input)" strokeWidth="6"
          strokeDasharray={`${arcLen} ${circumference}`}
          strokeLinecap="round"
          transform={`rotate(135 ${c} ${c})`}
        />
        <circle
          cx={c} cy={c} r={r} fill="none"
          stroke={color} strokeWidth="6"
          strokeDasharray={`${fillLen} ${circumference}`}
          strokeLinecap="round"
          transform={`rotate(135 ${c} ${c})`}
          style={{ animation: 'v7-regime-pulse 3s ease-in-out infinite' }}
        />
        <circle
          cx={c} cy={c} r={r} fill="none"
          stroke={color} strokeWidth="12"
          strokeDasharray={`${fillLen} ${circumference}`}
          strokeLinecap="round"
          transform={`rotate(135 ${c} ${c})`}
          opacity="0.12"
          style={{ filter: 'blur(3px)' }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-sm font-bold" style={{ color }}>{label}</span>
        <span className="text-[9px]" style={{ color: 'var(--text-muted)' }}>Regime</span>
      </div>
    </div>
  );
}

/** P&L Calendar Heatmap */
function PnLCalendar({
  data,
  currentDay,
}: {
  data: Record<number, number>;
  currentDay: number;
}) {
  const dayNames = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  const weeks: number[][] = [
    [1, 2, 3, 4, 5, 6, 7],
    [8, 9, 10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19, 20, 21],
  ];

  const maxPnL = Math.max(...Object.values(data).map(Math.abs), 1);

  const getColor = (day: number) => {
    const val = data[day];
    if (val === undefined) return 'transparent';
    if (val === 0) return 'var(--bg-input)';
    const intensity = Math.min(Math.abs(val) / maxPnL, 1);
    if (val > 0) return `color-mix(in srgb, var(--green) ${Math.round(intensity * 80 + 20)}%, var(--bg-input))`;
    return `color-mix(in srgb, var(--red) ${Math.round(intensity * 80 + 20)}%, var(--bg-input))`;
  };

  const tradingDays = Object.values(data).filter(v => v !== 0);
  const winDays = tradingDays.filter(v => v > 0).length;
  const consistency = tradingDays.length > 0 ? Math.round((winDays / tradingDays.length) * 100) : 0;

  return (
    <div>
      <div className="grid grid-cols-7 gap-1 mb-1">
        {dayNames.map(d => (
          <div key={d} className="text-center text-[9px]" style={{ color: 'var(--text-muted)' }}>{d}</div>
        ))}
      </div>
      {weeks.map((week, wi) => (
        <div key={wi} className="grid grid-cols-7 gap-1 mb-1">
          {week.map(day => {
            const val = data[day];
            const isToday = day === currentDay;
            const isWeekend = day % 7 === 6 || day % 7 === 0;
            return (
              <div
                key={day}
                className="aspect-square rounded-md flex flex-col items-center justify-center"
                style={{
                  background: isWeekend ? 'var(--bg-input)' : getColor(day),
                  border: isToday ? '2px solid var(--accent)' : '1px solid transparent',
                  opacity: isWeekend ? 0.3 : 1,
                }}
                title={val !== undefined ? `Day ${day}: ${val >= 0 ? '+' : ''}$${val}` : `Day ${day}`}
              >
                <span className="text-[9px] font-medium" style={{ color: 'var(--text-secondary)' }}>{day}</span>
                {val !== undefined && val !== 0 && !isWeekend && (
                  <span className="text-[7px] font-mono" style={{ color: val > 0 ? 'var(--green)' : 'var(--red)' }}>
                    {val > 0 ? '+' : ''}{val}
                  </span>
                )}
              </div>
            );
          })}
        </div>
      ))}
      <div className="flex justify-between mt-2 pt-2 border-t" style={{ borderColor: 'var(--border)' }}>
        <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
          Consistency: <span style={{ color: consistency >= 50 ? 'var(--green)' : 'var(--red)' }}>{consistency}%</span> win days
        </span>
        <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
          {winDays}W / {tradingDays.length - winDays}L
        </span>
      </div>
    </div>
  );
}

/** Goal Progress Bar */
function GoalProgress({
  current,
  target,
}: {
  current: number;
  target: number;
}) {
  const pct = Math.min((current / target) * 100, 100);
  const dayOfMonth = 22;
  const daysInMonth = 31;
  const daysRemaining = daysInMonth - dayOfMonth;
  const onTrack = pct >= (dayOfMonth / daysInMonth) * 100;
  const dailyNeeded = daysRemaining > 0 ? Math.round((target - current) / daysRemaining) : 0;

  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>March Target</span>
        <span className="text-xs font-mono" style={{ color: onTrack ? 'var(--green)' : 'var(--orange)' }}>
          {pct.toFixed(0)}%
        </span>
      </div>
      <div className="relative h-2.5 rounded-full overflow-hidden" style={{ background: 'var(--bg-input)' }}>
        <div
          className="h-full rounded-full"
          style={{
            width: `${pct}%`,
            background: onTrack ? 'var(--green)' : 'var(--orange)',
            animation: 'v7-progress-fill 1s ease-out',
            boxShadow: `0 0 8px ${onTrack ? 'rgba(76,175,80,0.4)' : 'rgba(255,152,0,0.4)'}`,
          }}
        />
        <div
          className="absolute top-0 h-full w-0.5"
          style={{
            left: `${(dayOfMonth / daysInMonth) * 100}%`,
            background: 'var(--text-muted)',
            opacity: 0.5,
          }}
        />
      </div>
      <div className="flex items-center justify-between mt-1.5">
        <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
          ${current.toLocaleString()} / ${target.toLocaleString()}
        </span>
        <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
          {onTrack ? 'On track' : `Need $${dailyNeeded}/day`}
        </span>
      </div>
    </div>
  );
}

/** Mini sparkline SVG for modal use */
function MiniSparkline({
  data,
  width = 200,
  height = 40,
  color = 'var(--green)',
}: {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
}) {
  const pad = 4;
  const chartW = width - pad * 2;
  const chartH = height - pad * 2;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  const points = data.map((v, i) => {
    const x = pad + (i / (data.length - 1)) * chartW;
    const y = pad + chartH - ((v - min) / range) * chartH;
    return `${x},${y}`;
  });

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <polyline
        points={points.join(' ')}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Detail Button Component — subtle expand icon for widget headers
// ---------------------------------------------------------------------------

function DetailButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="w-5 h-5 rounded flex items-center justify-center transition-colors flex-shrink-0"
      style={{ color: 'var(--text-muted)', background: 'transparent' }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = 'var(--bg-input)';
        e.currentTarget.style.color = 'var(--text-secondary)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = 'transparent';
        e.currentTarget.style.color = 'var(--text-muted)';
      }}
      title="View details"
    >
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M7 17L17 7" />
        <path d="M7 7h10v10" />
      </svg>
    </button>
  );
}

// ---------------------------------------------------------------------------
// Modal Section Component — consistent styling for sections inside modals
// ---------------------------------------------------------------------------

function ModalSection({ title, children }: { title?: string; children: React.ReactNode }) {
  return (
    <div
      className="rounded-lg p-4 mb-3 last:mb-0"
      style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}
    >
      {title && (
        <h4 className="text-xs font-semibold mb-2.5" style={{ color: 'var(--text-muted)' }}>{title}</h4>
      )}
      {children}
    </div>
  );
}

// Stat row helper for modals
function StatRow({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b last:border-0" style={{ borderColor: 'var(--border)' }}>
      <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{label}</span>
      <span className="text-xs font-mono font-medium" style={{ color: color || 'var(--text-primary)' }}>{value}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Dropdown Popover Hook
// ---------------------------------------------------------------------------

function usePopover() {
  const [isOpen, setIsOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    }
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen]);

  return { isOpen, setIsOpen, ref };
}

// ---------------------------------------------------------------------------
// Health helper functions (extracted for reuse in modals)
// ---------------------------------------------------------------------------

function sdColor(sd: number) {
  const abs = Math.abs(sd);
  if (abs <= 1) return 'var(--green)';
  if (abs <= 2) return 'var(--orange)';
  return 'var(--red)';
}

function sdBg(sd: number) {
  const abs = Math.abs(sd);
  if (abs <= 1) return 'var(--green-muted)';
  if (abs <= 2) return 'var(--orange-muted)';
  return 'var(--red-muted)';
}

function sdLabel(sd: number) {
  const abs = Math.abs(sd);
  if (abs <= 1) return 'On Track';
  if (abs <= 2) return 'Warning';
  return 'Critical';
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function DashboardV7() {
  const [showCustomize, setShowCustomize] = useState(false);
  const [customizeTab, setCustomizeTab] = useState<'KPIs' | 'Widgets' | 'Quick Actions'>('Widgets');
  const [healthViewTab, setHealthViewTab] = useState<'portfolios' | 'strategies'>('portfolios');
  const [widgets, setWidgets] = useState<WidgetConfig[]>(DEFAULT_WIDGETS);
  const [kpis, setKpis] = useState<KpiConfig[]>(DEFAULT_KPIS);
  const [portfolios, setPortfolios] = useState<PortfolioOption[]>(INITIAL_PORTFOLIOS);

  // V7: Detail modal states
  const [showEquityDetail, setShowEquityDetail] = useState(false);
  const [showPnLDetail, setShowPnLDetail] = useState(false);
  const [showCalendarDetail, setShowCalendarDetail] = useState(false);
  const [showPositionDetail, setShowPositionDetail] = useState<string | null>(null); // position ID
  const [showHealthDetail, setShowHealthDetail] = useState<string | null>(null); // health item ID
  const [showMarketDetail, setShowMarketDetail] = useState(false);
  const [showGoalDetail, setShowGoalDetail] = useState(false);
  const [showIssuesDetail, setShowIssuesDetail] = useState(false);
  const [showActivityDetail, setShowActivityDetail] = useState(false);
  const [activityFilter, setActivityFilter] = useState<'all' | 'entry' | 'exit' | 'system' | 'summary'>('all');
  const [activitySearch, setActivitySearch] = useState('');

  const portfolioPopover = usePopover();
  const kpiPopover = usePopover();

  const isEnabled = (id: string) => widgets.find(w => w.id === id)?.enabled ?? true;

  const toggleWidget = (id: string) => {
    setWidgets(prev => prev.map(w => w.id === id ? { ...w, enabled: !w.enabled } : w));
  };

  const toggleKpi = (id: string) => {
    setKpis(prev => prev.map(k => k.id === id ? { ...k, enabled: !k.enabled } : k));
  };

  const togglePortfolio = (id: string) => {
    setPortfolios(prev => prev.map(p => p.id === id ? { ...p, selected: !p.selected } : p));
  };

  const enabledKpis = kpis.filter(k => k.enabled);
  const selectedPortfolioCount = portfolios.filter(p => p.selected).length;

  const systemHealthy = MOCK_SYSTEM.polygon && MOCK_SYSTEM.ralph;
  const issueCount = MOCK_ISSUES.length;

  const alertTypeStyles: Record<string, { bg: string; color: string; icon: string }> = {
    entry: { bg: 'var(--green-muted)', color: 'var(--green)', icon: '\u2191' },
    exit: { bg: 'var(--red-muted)', color: 'var(--red)', icon: '\u2193' },
    system: { bg: 'rgba(156,163,175,0.15)', color: 'var(--text-muted)', icon: 'S' },
    summary: { bg: 'rgba(99,102,241,0.15)', color: 'var(--accent)', icon: '\u03A3' },
  };

  const severityStyles: Record<string, { bg: string; color: string; icon: string }> = {
    warning: { bg: 'var(--orange-muted)', color: 'var(--orange)', icon: '!' },
    error: { bg: 'var(--red-muted)', color: 'var(--red)', icon: '\u00D7' },
    info: { bg: 'var(--blue-muted)', color: 'var(--blue)', icon: 'i' },
  };

  // Determine KPI grid columns based on count
  const kpiGridCols = enabledKpis.length <= 4 ? 'grid-cols-4' :
    enabledKpis.length === 5 ? 'grid-cols-5' : 'grid-cols-6';

  // Filtered activity for activity modal
  const filteredActivity = MOCK_FULL_ACTIVITY
    .filter(a => activityFilter === 'all' || a.type === activityFilter)
    .filter(a => !activitySearch || a.message.toLowerCase().includes(activitySearch.toLowerCase()));

  return (
    <div style={{ animation: 'v7-fade-in 0.3s ease-out' }}>
      <style>{ANIMATION_STYLES}</style>

      {/* ============ HEADER: Title + Portfolio Filter + System Status + Customize ============ */}
      <div className="flex items-start justify-between mb-5">
        <div className="flex items-center gap-4">
          <div>
            <h1 className="text-2xl font-bold mb-1">Dashboard</h1>
            <span className="text-sm" style={{ color: 'var(--text-muted)' }}>
              Trading Cockpit
            </span>
          </div>

          {/* Portfolio Filter Dropdown */}
          <div className="relative" ref={portfolioPopover.ref}>
            <button
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors mt-1"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
              onClick={() => portfolioPopover.setIsOpen(!portfolioPopover.isOpen)}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z" />
              </svg>
              {selectedPortfolioCount} of {portfolios.length} portfolios
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M6 9l6 6 6-6" />
              </svg>
            </button>

            {portfolioPopover.isOpen && (
              <div
                className="absolute top-full left-0 mt-1 rounded-lg border py-1 z-50 min-w-[200px]"
                style={{ background: 'var(--bg-card)', borderColor: 'var(--border)', boxShadow: '0 8px 24px rgba(0,0,0,0.4)' }}
              >
                <div className="px-3 py-1.5 border-b" style={{ borderColor: 'var(--border)' }}>
                  <span className="text-[10px] font-medium" style={{ color: 'var(--text-muted)' }}>
                    Filter dashboard by portfolio
                  </span>
                </div>
                {portfolios.map(p => (
                  <label
                    key={p.id}
                    className="flex items-center gap-2.5 px-3 py-2 cursor-pointer transition-colors"
                    style={{ color: 'var(--text-secondary)' }}
                    onMouseEnter={e => (e.currentTarget.style.background = 'var(--bg-input)')}
                    onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
                  >
                    <input
                      type="checkbox"
                      checked={p.selected}
                      onChange={() => togglePortfolio(p.id)}
                      className="w-3.5 h-3.5 rounded accent-[var(--accent)]"
                    />
                    <span className="text-xs">{p.name}</span>
                  </label>
                ))}
              </div>
            )}
          </div>

          {/* System Status — compact inline indicator */}
          <div
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[11px] mt-1 cursor-default"
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
            title={`Polygon: ${MOCK_SYSTEM.polygon ? 'Connected' : 'Down'} | Ralph: ${MOCK_SYSTEM.ralph ? 'Connected' : 'Down'} | Uptime: ${MOCK_SYSTEM.workerUptime}`}
          >
            <span
              className="w-2 h-2 rounded-full flex-shrink-0"
              style={{
                background: systemHealthy ? 'var(--green)' : 'var(--red)',
                boxShadow: systemHealthy ? '0 0 6px rgba(76,175,80,0.5)' : '0 0 6px rgba(244,67,54,0.5)',
                animation: 'v7-pulse-dot 2s ease-in-out infinite',
              }}
            />
            <span style={{ color: systemHealthy ? 'var(--green)' : 'var(--red)' }}>
              {systemHealthy ? 'All systems operational' : `${issueCount} issue${issueCount !== 1 ? 's' : ''}`}
            </span>
          </div>

          {/* Quick Action Buttons — compact icons */}
          <div className="flex items-center gap-1 mt-1">
            {[
              { label: 'Strategy Builder', href: '/strategy-builder', icon: 'M12 4v16m-8-8h16' },
              { label: 'Portfolios', href: '/portfolios', icon: 'M4 6h16M4 12h16M4 18h16' },
              { label: 'Alerts', href: '/alerts', icon: 'M18 8A6 6 0 006 8c0 7-3 9-3 9h18s-3-2-3-9M13.73 21a2 2 0 01-3.46 0' },
              { label: 'Settings', href: '/settings', icon: 'M12 15a3 3 0 100-6 3 3 0 000 6z' },
            ].map((action) => (
              <Link key={action.label} href={action.href}>
                <div
                  className="w-8 h-8 rounded-lg flex items-center justify-center transition-colors"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
                  title={action.label}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d={action.icon} />
                  </svg>
                </div>
              </Link>
            ))}
          </div>
        </div>

        <button
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors"
          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
          onClick={() => setShowCustomize(true)}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <circle cx="12" cy="12" r="3" />
            <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z" />
          </svg>
          Customize
        </button>
      </div>

      {/* ============ CUSTOMIZABLE KPI STRIP ============ */}
      <div className="flex items-center gap-2 mb-5">
        <div className={`grid ${kpiGridCols} gap-3 flex-1`}>
          {enabledKpis.map(kpi => (
            <div
              key={kpi.id}
              className="rounded-lg border p-3"
              style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}
            >
              <p className="text-[10px] mb-0.5" style={{ color: 'var(--text-muted)' }}>{kpi.label}</p>
              <div className="flex items-baseline gap-1.5">
                <span
                  className="text-lg font-semibold"
                  style={{
                    color: kpi.id === 'today-pnl' && kpi.positive ? 'var(--green)' :
                      kpi.id === 'market-regime' ? 'var(--green)' :
                      kpi.id === 'max-dd' ? 'var(--red)' :
                      'var(--text-primary)',
                  }}
                >
                  {kpi.value}
                </span>
                {kpi.delta && (
                  <span
                    className="text-xs font-mono"
                    style={{ color: kpi.positive ? 'var(--green)' : 'var(--red)' }}
                  >
                    {kpi.delta}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>

      </div>

      {/* ============ MAIN CONTENT: Two-column layout (7/5 split — V5-style spacing) ============ */}
      <div className="grid grid-cols-12 gap-5 mb-5">
        {/* -- LEFT COLUMN: Charts (7 cols ~ 58%) -- */}
        <div className="col-span-7 space-y-5">
          {/* Portfolio Equity Curve */}
          {isEnabled('equity-curve') && (
            <Card>
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>
                  Portfolio Equity Curve
                </h3>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-mono" style={{ color: 'var(--green)' }}>
                    +${EQUITY_DATA[EQUITY_DATA.length - 1].toLocaleString()} cumulative
                  </span>
                  <DetailButton onClick={() => setShowEquityDetail(true)} />
                </div>
              </div>
              <EquityCurve data={EQUITY_DATA} width={700} height={260} />
            </Card>
          )}

          {/* Daily P&L Bar Chart */}
          {isEnabled('daily-pnl') && (
            <Card>
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>
                  Daily P&L
                </h3>
                <div className="flex items-center gap-2">
                  <div className="flex items-center gap-3">
                    <div className="flex items-center gap-1">
                      <span className="w-2 h-2 rounded-sm" style={{ background: 'var(--green)' }} />
                      <span className="text-[9px]" style={{ color: 'var(--text-muted)' }}>Profit</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <span className="w-2 h-2 rounded-sm" style={{ background: 'var(--red)' }} />
                      <span className="text-[9px]" style={{ color: 'var(--text-muted)' }}>Loss</span>
                    </div>
                  </div>
                  <DetailButton onClick={() => setShowPnLDetail(true)} />
                </div>
              </div>
              <DailyPnLChart data={DAILY_PNL} width={700} height={220} />
            </Card>
          )}

          {/* P&L Calendar Heatmap — under Daily P&L */}
          {isEnabled('calendar') && (
            <Card>
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>March 2026 P&L</h3>
                <div className="flex items-center gap-2">
                  <div className="flex items-center gap-2">
                    <div className="flex items-center gap-1">
                      <span className="w-2 h-2 rounded-sm" style={{ background: 'var(--green)' }} />
                      <span className="text-[9px]" style={{ color: 'var(--text-muted)' }}>Profit</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <span className="w-2 h-2 rounded-sm" style={{ background: 'var(--red)' }} />
                      <span className="text-[9px]" style={{ color: 'var(--text-muted)' }}>Loss</span>
                    </div>
                  </div>
                  <DetailButton onClick={() => setShowCalendarDetail(true)} />
                </div>
              </div>
              <PnLCalendar data={CALENDAR_DATA} currentDay={22} />
            </Card>
          )}

        </div>

        {/* -- RIGHT COLUMN: Real-time widgets (5 cols ~ 42%) -- */}
        <div className="col-span-5 space-y-5">
          {/* Active Positions — top of right column */}
          {isEnabled('positions') && (
            <Card>
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>
                  Active Positions ({MOCK_POSITIONS.length})
                </h3>
                <span
                  className="flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full"
                  style={{ background: 'var(--green-muted)', color: 'var(--green)' }}
                >
                  <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--green)' }} />
                  Live
                </span>
              </div>
              {MOCK_POSITIONS.map((pos, idx) => (
                <div
                  key={pos.id}
                  className="rounded-lg p-3 mb-2 last:mb-0"
                  style={{
                    background: 'var(--bg-input)',
                    border: '1px solid var(--border)',
                    animation: `v7-glow-border 3s ease-in-out infinite, v7-slide-in 0.3s ease-out ${idx * 0.1}s both`,
                  }}
                >
                  {/* Top row: symbol, direction, match status, detail + close button */}
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-semibold">{pos.symbol}</span>
                      <span
                        className="text-[10px] px-1.5 py-0.5 rounded"
                        style={{
                          background: pos.direction === 'LONG' ? 'var(--green-muted)' : 'var(--red-muted)',
                          color: pos.direction === 'LONG' ? 'var(--green)' : 'var(--red)',
                        }}
                      >
                        {pos.direction}
                      </span>
                      <span
                        className="text-[9px] px-1.5 py-0.5 rounded-full font-medium"
                        style={{
                          background: pos.matched ? 'var(--green-muted)' : 'var(--orange-muted)',
                          color: pos.matched ? 'var(--green)' : 'var(--orange)',
                        }}
                      >
                        {pos.matched ? 'Matched' : 'Anomaly'}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span
                        className="text-sm font-mono font-bold"
                        style={{ color: pos.unrealizedR >= 0 ? 'var(--green)' : 'var(--red)' }}
                      >
                        {pos.unrealizedR >= 0 ? '+' : ''}{pos.unrealizedR.toFixed(2)}R
                      </span>
                      <DetailButton onClick={() => setShowPositionDetail(pos.id)} />
                      <button
                        className="text-[9px] px-2 py-0.5 rounded font-medium transition-opacity hover:opacity-80"
                        style={{ background: 'var(--red-muted)', color: 'var(--red)' }}
                        title="Close position early"
                      >
                        Close
                      </button>
                    </div>
                  </div>

                  {/* Detail row */}
                  <div className="grid grid-cols-3 gap-2 text-[10px]">
                    <div>
                      <span style={{ color: 'var(--text-muted)' }}>Entry</span>
                      <p className="font-mono" style={{ color: 'var(--text-secondary)' }}>${pos.entryPrice.toFixed(2)}</p>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)' }}>Current</span>
                      <p className="font-mono" style={{ color: 'var(--text-secondary)' }}>${pos.currentPrice.toFixed(2)}</p>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)' }}>Since</span>
                      <p style={{ color: 'var(--text-secondary)' }}>{pos.entryTime}</p>
                    </div>
                  </div>

                  <p className="text-[10px] mt-1.5" style={{ color: 'var(--text-muted)' }}>{pos.strategy}</p>
                </div>
              ))}
              {MOCK_POSITIONS.length === 0 && (
                <p className="text-xs text-center py-4" style={{ color: 'var(--text-muted)' }}>No open positions</p>
              )}
            </Card>
          )}

          {/* Portfolio Health — SD deviation widget with tabs */}
          {isEnabled('health') && (() => {
            const [healthTab, setHealthTab] = [healthViewTab, setHealthViewTab];
            const items = healthTab === 'portfolios' ? MOCK_HEALTH_PORTFOLIOS : MOCK_HEALTH_STRATEGIES;

            return (
              <Card>
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>Performance Health</h3>
                  <div className="flex gap-0.5 rounded-lg overflow-hidden" style={{ border: '1px solid var(--border)' }}>
                    {([{ id: 'portfolios' as const, label: 'Portfolios' }, { id: 'strategies' as const, label: 'Strategies' }]).map((t) => (
                      <button
                        key={t.id}
                        onClick={() => setHealthTab(t.id)}
                        className="px-2.5 py-1 text-[10px] font-medium"
                        style={{
                          background: healthTab === t.id ? 'var(--accent-muted)' : 'transparent',
                          color: healthTab === t.id ? 'var(--accent)' : 'var(--text-muted)',
                        }}
                      >
                        {t.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* SD legend */}
                <div className="flex items-center gap-3 mb-3 px-2 py-1.5 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                  <div className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full" style={{ background: 'var(--green)' }} />
                    <span className="text-[9px]" style={{ color: 'var(--text-muted)' }}>&lt;1 SD</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full" style={{ background: 'var(--orange)' }} />
                    <span className="text-[9px]" style={{ color: 'var(--text-muted)' }}>1-2 SD</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full" style={{ background: 'var(--red)' }} />
                    <span className="text-[9px]" style={{ color: 'var(--text-muted)' }}>&gt;2 SD</span>
                  </div>
                </div>

                {/* Items */}
                <div className="space-y-2">
                  {items.map((item) => (
                    <div
                      key={item.id}
                      className="rounded-lg p-2.5"
                      style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
                    >
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="text-xs font-medium truncate flex-1 mr-2" style={{ color: 'var(--text-primary)' }}>{item.name}</span>
                        <div className="flex items-center gap-1.5 flex-shrink-0">
                          <span
                            className="text-[9px] px-1.5 py-0.5 rounded-full font-medium"
                            style={{ background: sdBg(item.deviationSd), color: sdColor(item.deviationSd) }}
                          >
                            {sdLabel(item.deviationSd)}
                          </span>
                          <DetailButton onClick={() => setShowHealthDetail(item.id)} />
                        </div>
                      </div>

                      {/* SD bar visualization */}
                      <div className="relative h-3 rounded-full mb-1.5" style={{ background: 'var(--bg-card)' }}>
                        {/* Zone backgrounds */}
                        <div className="absolute inset-y-0 rounded-full" style={{ left: '10%', right: '10%', background: 'var(--green-muted)', opacity: 0.4 }} />
                        <div className="absolute inset-y-0 rounded-full" style={{ left: '3%', width: '7%', background: 'var(--orange-muted)', opacity: 0.4 }} />
                        <div className="absolute inset-y-0 rounded-full" style={{ right: '3%', width: '7%', background: 'var(--orange-muted)', opacity: 0.4 }} />

                        {/* Center line (expected) */}
                        <div className="absolute top-0 bottom-0 w-px" style={{ left: '50%', background: 'var(--text-muted)', opacity: 0.4 }} />

                        {/* Marker — position based on SD (-3 to +3 mapped to 0% to 100%) */}
                        <div
                          className="absolute top-0 bottom-0 w-2.5 rounded-full"
                          style={{
                            left: `${Math.max(2, Math.min(98, 50 + (item.deviationSd / 3) * 45))}%`,
                            transform: 'translateX(-50%)',
                            background: sdColor(item.deviationSd),
                            boxShadow: `0 0 4px ${sdColor(item.deviationSd)}`,
                          }}
                        />
                      </div>

                      {/* Stats */}
                      <div className="flex items-center justify-between text-[10px]">
                        <span style={{ color: 'var(--text-muted)' }}>{item.alertTrades} trades</span>
                        <span className="font-mono" style={{ color: sdColor(item.deviationSd) }}>
                          {item.deviationSd >= 0 ? '+' : ''}{item.deviationSd.toFixed(2)} SD
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </Card>
            );
          })()}

          {/* Market Regime */}
          {isEnabled('market-regime') && (
            <Card>
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>Market Regime</h3>
                <DetailButton onClick={() => setShowMarketDetail(true)} />
              </div>
              <div className="flex items-center gap-4">
                <RegimeGauge regime={MOCK_MARKET.regime} size={90} />
                <div className="flex-1 space-y-1.5">
                  <div className="flex items-center justify-between text-xs">
                    <span style={{ color: 'var(--text-muted)' }}>VIX</span>
                    <span className="font-mono" style={{ color: MOCK_MARKET.vix < 20 ? 'var(--green)' : 'var(--red)' }}>
                      {MOCK_MARKET.vix.toFixed(1)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span style={{ color: 'var(--text-muted)' }}>Breadth</span>
                    <span className="font-mono" style={{ color: MOCK_MARKET.breadth.ratio > 1 ? 'var(--green)' : 'var(--red)' }}>
                      {MOCK_MARKET.breadth.ratio.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span style={{ color: 'var(--text-muted)' }}>SPY</span>
                    <span className="font-mono">
                      {MOCK_MARKET.spy.price}{' '}
                      <span style={{ color: MOCK_MARKET.spy.changePct >= 0 ? 'var(--green)' : 'var(--red)' }}>
                        {MOCK_MARKET.spy.changePct >= 0 ? '+' : ''}{MOCK_MARKET.spy.changePct.toFixed(2)}%
                      </span>
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span style={{ color: 'var(--text-muted)' }}>QQQ</span>
                    <span className="font-mono">
                      {MOCK_MARKET.qqq.price}{' '}
                      <span style={{ color: MOCK_MARKET.qqq.changePct >= 0 ? 'var(--green)' : 'var(--red)' }}>
                        {MOCK_MARKET.qqq.changePct >= 0 ? '+' : ''}{MOCK_MARKET.qqq.changePct.toFixed(2)}%
                      </span>
                    </span>
                  </div>
                </div>
              </div>
            </Card>
          )}

          {/* Monthly Goal Tracker */}
          {isEnabled('monthly-goal') && (
            <Card>
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>Monthly Goal</h3>
                <DetailButton onClick={() => setShowGoalDetail(true)} />
              </div>
              <GoalProgress
                current={MOCK_PORTFOLIO.monthlyProgress}
                target={MOCK_PORTFOLIO.monthlyTarget}
              />
            </Card>
          )}
        </div>
      </div>

      {/* ============ BOTTOM ROW: Issues + Activity (half/half) ============ */}
      <div className="grid grid-cols-2 gap-5 mb-5">
          {/* Issues & Warnings */}
          {isEnabled('issues') && (
            <Card>
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>
                  Issues & Warnings
                </h3>
                <div className="flex items-center gap-2">
                  <span
                    className="text-[9px] px-1.5 py-0.5 rounded-full font-medium"
                    style={{
                      background: MOCK_ISSUES.length > 0 ? 'var(--orange-muted)' : 'var(--green-muted)',
                      color: MOCK_ISSUES.length > 0 ? 'var(--orange)' : 'var(--green)',
                    }}
                  >
                    {MOCK_ISSUES.length} active
                  </span>
                  <DetailButton onClick={() => setShowIssuesDetail(true)} />
                </div>
              </div>
              <div className="space-y-0">
                {MOCK_ISSUES.map((issue) => {
                  const style = severityStyles[issue.severity] || severityStyles.info;
                  return (
                    <Link key={issue.id} href={issue.link}>
                      <div
                        className="flex items-start gap-2.5 py-2.5 border-b last:border-0 transition-colors cursor-pointer"
                        style={{ borderColor: 'var(--border)' }}
                      >
                        <div
                          className="w-5 h-5 rounded flex items-center justify-center text-[10px] font-bold flex-shrink-0 mt-0.5"
                          style={{ background: style.bg, color: style.color }}
                        >
                          {style.icon}
                        </div>
                        <p className="text-xs flex-1 leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                          {issue.message}
                        </p>
                        <span
                          className="text-[9px] px-1.5 py-0.5 rounded flex-shrink-0 mt-0.5"
                          style={{ background: style.bg, color: style.color }}
                        >
                          {issue.severity}
                        </span>
                      </div>
                    </Link>
                  );
                })}
                {MOCK_ISSUES.length === 0 && (
                  <p className="text-xs text-center py-4" style={{ color: 'var(--text-muted)' }}>No active issues</p>
                )}
              </div>
            </Card>
          )}

          {/* Recent Activity Feed */}
          {isEnabled('activity') && (
            <Card>
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>Recent Activity</h3>
                <DetailButton onClick={() => setShowActivityDetail(true)} />
              </div>
              <div className="space-y-0">
                {MOCK_ALERTS.map((alert, idx) => {
                  const typeStyle = alertTypeStyles[alert.type] || alertTypeStyles.system;
                  return (
                    <div
                      key={idx}
                      className="flex items-center gap-2.5 py-2 border-b last:border-0"
                      style={{ borderColor: 'var(--border)' }}
                    >
                      <div
                        className="w-5 h-5 rounded flex items-center justify-center text-[10px] font-bold flex-shrink-0"
                        style={{ background: typeStyle.bg, color: typeStyle.color }}
                      >
                        {typeStyle.icon}
                      </div>
                      <p className="text-xs flex-1 truncate" style={{ color: 'var(--text-secondary)' }}>
                        {alert.message}
                      </p>
                      <span className="text-[10px] flex-shrink-0" style={{ color: 'var(--text-muted)' }}>
                        {alert.time}
                      </span>
                    </div>
                  );
                })}
              </div>
            </Card>
          )}
      </div>

      {/* Quick actions are now in the header row */}

      {/* ============ CUSTOMIZE MODAL (Tabbed: KPIs / Widgets / Quick Actions) ============ */}
      <Modal
        title="Customize Dashboard"
        isOpen={showCustomize}
        onClose={() => setShowCustomize(false)}
        width="520px"
      >
        {/* Modal Tabs */}
        <div className="flex gap-1 border-b mb-4" style={{ borderColor: 'var(--border)' }}>
          {(['KPIs', 'Widgets', 'Quick Actions'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setCustomizeTab(tab)}
              className="px-4 py-2 text-xs font-medium transition-colors"
              style={{
                color: customizeTab === tab ? 'var(--accent)' : 'var(--text-muted)',
                borderBottom: customizeTab === tab ? '2px solid var(--accent)' : '2px solid transparent',
                marginBottom: '-1px',
              }}
            >
              {tab}
            </button>
          ))}
        </div>

        {/* KPIs Tab */}
        {customizeTab === 'KPIs' && (
          <div>
            <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>Choose which KPIs appear in the top strip (4-6 recommended).</p>
            <div className="space-y-0">
              {kpis.map(kpi => (
                <label
                  key={kpi.id}
                  className="flex items-center gap-2.5 py-2.5 border-b last:border-0 cursor-pointer"
                  style={{ borderColor: 'var(--border)' }}
                >
                  <input type="checkbox" checked={kpi.enabled} onChange={() => toggleKpi(kpi.id)} className="w-3.5 h-3.5 rounded accent-[var(--accent)]" />
                  <span className="text-sm flex-1" style={{ color: 'var(--text-secondary)' }}>{kpi.label}</span>
                  <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>{kpi.value}</span>
                </label>
              ))}
            </div>
          </div>
        )}

        {/* Widgets Tab */}
        {customizeTab === 'Widgets' && (
          <div>
            <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>Toggle dashboard widgets on or off. Widget sizing and positioning coming soon.</p>
            <div className="space-y-0">
              {widgets.map(widget => (
                <div
                  key={widget.id}
                  className="flex items-center justify-between py-2.5 border-b last:border-0"
                  style={{ borderColor: 'var(--border)' }}
                >
                  <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>{widget.label}</span>
                  <button
                    className="relative w-10 h-5 rounded-full transition-colors"
                    style={{ background: widget.enabled ? 'var(--accent)' : 'var(--bg-input)', border: '1px solid', borderColor: widget.enabled ? 'var(--accent)' : 'var(--border)' }}
                    onClick={() => toggleWidget(widget.id)}
                  >
                    <span className="absolute top-0.5 w-3.5 h-3.5 rounded-full transition-all" style={{ background: widget.enabled ? '#000' : 'var(--text-muted)', left: widget.enabled ? '22px' : '3px' }} />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Quick Actions Tab */}
        {customizeTab === 'Quick Actions' && (
          <div>
            <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>Choose which quick action buttons appear in the header bar.</p>
            <div className="space-y-0">
              {[
                { id: 'strategy-builder', label: 'Strategy Builder', icon: 'M12 4v16m-8-8h16', enabled: true },
                { id: 'portfolios', label: 'Portfolios', icon: 'M4 6h16M4 12h16M4 18h16', enabled: true },
                { id: 'alerts', label: 'Alerts', icon: 'M18 8A6 6 0 006 8c0 7-3 9-3 9h18s-3-2-3-9M13.73 21a2 2 0 01-3.46 0', enabled: true },
                { id: 'settings', label: 'Settings', icon: 'M12 15a3 3 0 100-6 3 3 0 000 6z', enabled: true },
                { id: 'mass-builder', label: 'Mass Builder', icon: 'M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z', enabled: false },
                { id: 'confluence', label: 'Confluence Packs', icon: 'M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5', enabled: false },
                { id: 'marketplace', label: 'Marketplace', icon: 'M3 3h18v18H3zM3 9h18M9 21V9', enabled: false },
                { id: 'requirements', label: 'Requirements', icon: 'M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2', enabled: false },
              ].map(action => (
                <label
                  key={action.id}
                  className="flex items-center gap-2.5 py-2.5 border-b last:border-0 cursor-pointer"
                  style={{ borderColor: 'var(--border)' }}
                >
                  <input type="checkbox" defaultChecked={action.enabled} className="w-3.5 h-3.5 rounded accent-[var(--accent)]" />
                  <div className="w-6 h-6 rounded flex items-center justify-center flex-shrink-0" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d={action.icon} />
                    </svg>
                  </div>
                  <span className="text-sm flex-1" style={{ color: 'var(--text-secondary)' }}>{action.label}</span>
                </label>
              ))}
            </div>
          </div>
        )}
      </Modal>

      {/* ============ V7 DETAIL MODALS ============ */}

      {/* 1. Equity Details Modal */}
      <Modal
        title="Equity Details"
        isOpen={showEquityDetail}
        onClose={() => setShowEquityDetail(false)}
        width="780px"
      >
        <ModalSection title="Equity Curve (Expanded)">
          <EquityCurve data={EQUITY_DATA} width={720} height={320} />
        </ModalSection>

        <ModalSection title="Segment Breakdown">
          <div className="grid grid-cols-3 gap-3">
            <div className="rounded-lg p-3 text-center" style={{ background: 'var(--bg-input)' }}>
              <p className="text-[10px] mb-1" style={{ color: 'var(--text-muted)' }}>Backtest Total R</p>
              <p className="text-lg font-semibold font-mono" style={{ color: 'var(--text-primary)' }}>+{MOCK_EQUITY_SEGMENTS.btTotalR}R</p>
            </div>
            <div className="rounded-lg p-3 text-center" style={{ background: 'var(--bg-input)' }}>
              <p className="text-[10px] mb-1" style={{ color: 'var(--text-muted)' }}>Forward Test Total R</p>
              <p className="text-lg font-semibold font-mono" style={{ color: 'var(--accent)' }}>+{MOCK_EQUITY_SEGMENTS.fwTotalR}R</p>
            </div>
            <div className="rounded-lg p-3 text-center" style={{ background: 'var(--bg-input)' }}>
              <p className="text-[10px] mb-1" style={{ color: 'var(--text-muted)' }}>Live Total R</p>
              <p className="text-lg font-semibold font-mono" style={{ color: 'var(--green)' }}>+{MOCK_EQUITY_SEGMENTS.liveTotalR}R</p>
            </div>
          </div>
        </ModalSection>

        <ModalSection title="Key Inflection Points">
          {MOCK_EQUITY_INFLECTIONS.map((pt, i) => (
            <div key={i} className="flex items-center justify-between py-2 border-b last:border-0" style={{ borderColor: 'var(--border)' }}>
              <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>{pt.label}</span>
              <div className="flex items-center gap-3">
                <span className="text-xs font-mono font-medium" style={{ color: pt.value.startsWith('+') ? 'var(--green)' : pt.value.startsWith('-') ? 'var(--red)' : 'var(--text-primary)' }}>
                  {pt.value}
                </span>
                <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{pt.date}</span>
              </div>
            </div>
          ))}
        </ModalSection>

        <ModalSection title="vs Last Month">
          <div className="flex items-center gap-3">
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Equity growth comparison</span>
            <span className="text-sm font-semibold" style={{ color: 'var(--green)' }}>+15% improvement</span>
          </div>
          <p className="text-[10px] mt-2" style={{ color: 'var(--text-muted)' }}>
            Last month ended at +$1,820 cumulative. Current month is at +$2,500 with 9 trading days remaining.
          </p>
        </ModalSection>

        <div className="mt-3">
          <Link href="/portfolios/p1" className="text-xs font-medium" style={{ color: 'var(--accent)' }}>
            View full portfolio detail page &rarr;
          </Link>
        </div>
      </Modal>

      {/* 2. P&L Breakdown Modal */}
      <Modal
        title="P&L Breakdown"
        isOpen={showPnLDetail}
        onClose={() => setShowPnLDetail(false)}
        width="680px"
      >
        <ModalSection title="Summary Stats">
          <div className="grid grid-cols-2 gap-x-6">
            <StatRow label="Total Trading Days" value={String(MOCK_PNL_STATS.totalDays)} />
            <StatRow label="Win Days" value={String(MOCK_PNL_STATS.winDays)} color="var(--green)" />
            <StatRow label="Loss Days" value={String(MOCK_PNL_STATS.lossDays)} color="var(--red)" />
            <StatRow label="Win Rate" value={`${Math.round((MOCK_PNL_STATS.winDays / MOCK_PNL_STATS.totalDays) * 100)}%`} />
            <StatRow label="Avg Win Day" value={`+$${MOCK_PNL_STATS.avgWinDay.toFixed(0)}`} color="var(--green)" />
            <StatRow label="Avg Loss Day" value={`-$${Math.abs(MOCK_PNL_STATS.avgLossDay).toFixed(0)}`} color="var(--red)" />
          </div>
        </ModalSection>

        <ModalSection title="Highlights">
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-lg p-3" style={{ background: 'var(--green-muted)' }}>
              <p className="text-[10px] mb-0.5" style={{ color: 'var(--text-muted)' }}>Best Day</p>
              <p className="text-sm font-semibold font-mono" style={{ color: 'var(--green)' }}>+${MOCK_PNL_STATS.bestDay.value}</p>
              <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{MOCK_PNL_STATS.bestDay.date}</p>
            </div>
            <div className="rounded-lg p-3" style={{ background: 'var(--red-muted)' }}>
              <p className="text-[10px] mb-0.5" style={{ color: 'var(--text-muted)' }}>Worst Day</p>
              <p className="text-sm font-semibold font-mono" style={{ color: 'var(--red)' }}>-${Math.abs(MOCK_PNL_STATS.worstDay.value)}</p>
              <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{MOCK_PNL_STATS.worstDay.date}</p>
            </div>
          </div>
        </ModalSection>

        <ModalSection title="Current Streak">
          <div className="flex items-center gap-2">
            <span
              className="text-xs px-2 py-1 rounded font-medium"
              style={{
                background: MOCK_PNL_STATS.currentStreak.type === 'win' ? 'var(--green-muted)' : 'var(--red-muted)',
                color: MOCK_PNL_STATS.currentStreak.type === 'win' ? 'var(--green)' : 'var(--red)',
              }}
            >
              {MOCK_PNL_STATS.currentStreak.days} {MOCK_PNL_STATS.currentStreak.type} days in a row
            </span>
          </div>
        </ModalSection>

        <ModalSection title="Weekly Breakdown">
          <div className="space-y-0">
            {MOCK_PNL_WEEKLY.map((week, i) => (
              <div key={i} className="flex items-center justify-between py-2.5 border-b last:border-0" style={{ borderColor: 'var(--border)' }}>
                <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>{week.week}</span>
                <div className="flex items-center gap-4">
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{week.trades} trades</span>
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{week.winRate}% WR</span>
                  <span className="text-xs font-mono font-medium" style={{ color: week.pnl >= 0 ? 'var(--green)' : 'var(--red)' }}>
                    {week.pnl >= 0 ? '+' : ''}${week.pnl}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </ModalSection>

        <div className="mt-3">
          <Link href="/portfolios/p1" className="text-xs font-medium" style={{ color: 'var(--accent)' }}>
            View portfolio performance tab &rarr;
          </Link>
        </div>
      </Modal>

      {/* 3. Monthly Summary Modal (Calendar) */}
      <Modal
        title="Monthly Summary \u2014 March 2026"
        isOpen={showCalendarDetail}
        onClose={() => setShowCalendarDetail(false)}
        width="680px"
      >
        <ModalSection title="Month Stats">
          <div className="grid grid-cols-2 gap-x-6">
            <StatRow
              label="Total P&L"
              value={`+$${Object.values(CALENDAR_DATA).reduce((a, b) => a + b, 0).toLocaleString()}`}
              color="var(--green)"
            />
            <StatRow label="Trading Days" value={String(Object.keys(CALENDAR_DATA).length)} />
            <StatRow
              label="Win Rate by Day"
              value={`${Math.round((Object.values(CALENDAR_DATA).filter(v => v > 0).length / Object.keys(CALENDAR_DATA).length) * 100)}%`}
            />
            <StatRow label="Avg P&L per Day" value={`$${Math.round(Object.values(CALENDAR_DATA).reduce((a, b) => a + b, 0) / Object.keys(CALENDAR_DATA).length)}`} />
          </div>
        </ModalSection>

        <ModalSection title="Week-by-Week">
          {MOCK_PNL_WEEKLY.map((week, i) => {
            const pct = (week.pnl / Math.max(...MOCK_PNL_WEEKLY.map(w => Math.abs(w.pnl)))) * 100;
            return (
              <div key={i} className="mb-2.5 last:mb-0">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>{week.week}</span>
                  <span className="text-xs font-mono font-medium" style={{ color: week.pnl >= 0 ? 'var(--green)' : 'var(--red)' }}>
                    {week.pnl >= 0 ? '+' : ''}${week.pnl}
                  </span>
                </div>
                <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--bg-input)' }}>
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${Math.abs(pct)}%`,
                      background: week.pnl >= 0 ? 'var(--green)' : 'var(--red)',
                    }}
                  />
                </div>
              </div>
            );
          })}
        </ModalSection>

        <ModalSection title="vs Previous Month (February)">
          <div className="grid grid-cols-2 gap-x-6">
            <StatRow label="Feb Total P&L" value={`$${MOCK_MONTHLY_SUMMARY.prevMonthPnL.toLocaleString()}`} />
            <StatRow label="Feb Win Rate" value={`${MOCK_MONTHLY_SUMMARY.prevMonthWinRate}%`} />
            <StatRow label="Feb Trading Days" value={String(MOCK_MONTHLY_SUMMARY.prevMonthTradingDays)} />
            <StatRow
              label="Change"
              value={`+${Math.round(((Object.values(CALENDAR_DATA).reduce((a, b) => a + b, 0) - MOCK_MONTHLY_SUMMARY.prevMonthPnL) / MOCK_MONTHLY_SUMMARY.prevMonthPnL) * 100)}%`}
              color="var(--green)"
            />
          </div>
        </ModalSection>

        <ModalSection title="Consistency Score">
          <div className="flex items-center gap-3 mb-2">
            <span className="text-2xl font-bold" style={{ color: MOCK_MONTHLY_SUMMARY.consistencyScore >= 60 ? 'var(--green)' : 'var(--orange)' }}>
              {MOCK_MONTHLY_SUMMARY.consistencyScore}
            </span>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>/ 100</span>
          </div>
          <p className="text-[11px] leading-relaxed" style={{ color: 'var(--text-muted)' }}>
            {MOCK_MONTHLY_SUMMARY.consistencyExplanation}
          </p>
        </ModalSection>

        <ModalSection title="Goal Progress">
          <GoalProgress current={MOCK_MONTHLY_SUMMARY.goalProgress.current} target={MOCK_MONTHLY_SUMMARY.goalProgress.target} />
        </ModalSection>
      </Modal>

      {/* 4. Position Detail Modal (per position) */}
      {showPositionDetail && (() => {
        const pos = MOCK_POSITIONS.find(p => p.id === showPositionDetail);
        const detail = MOCK_POSITION_DETAILS[showPositionDetail];
        if (!pos || !detail) return null;

        const stopDist = Math.abs(pos.currentPrice - detail.stopLevel);
        const targetDist = Math.abs(detail.targetLevel - pos.currentPrice);

        return (
          <Modal
            title={`Position Detail \u2014 ${pos.symbol} ${pos.direction}`}
            isOpen={true}
            onClose={() => setShowPositionDetail(null)}
            width="680px"
          >
            <ModalSection title="Trade Context">
              <StatRow label="Strategy" value={pos.strategy} />
              <StatRow label="Entry Trigger" value={detail.entryTrigger} />
              <StatRow label="Entry Price" value={`$${pos.entryPrice.toFixed(2)}`} />
              <StatRow label="Current Price" value={`$${pos.currentPrice.toFixed(2)}`} color={pos.unrealizedR >= 0 ? 'var(--green)' : 'var(--red)'} />
              <StatRow label="Unrealized R" value={`${pos.unrealizedR >= 0 ? '+' : ''}${pos.unrealizedR.toFixed(2)}R`} color={pos.unrealizedR >= 0 ? 'var(--green)' : 'var(--red)'} />
            </ModalSection>

            <ModalSection title="Confluence Conditions at Entry">
              <div className="flex flex-wrap gap-1.5">
                {detail.confluenceConditions.map((cond, i) => (
                  <span
                    key={i}
                    className="text-[10px] px-2 py-1 rounded font-mono"
                    style={{ background: 'var(--green-muted)', color: 'var(--green)' }}
                  >
                    {cond}
                  </span>
                ))}
              </div>
            </ModalSection>

            <ModalSection title="Stop & Target Levels">
              <div className="grid grid-cols-2 gap-3 mb-2">
                <div className="rounded-lg p-3" style={{ background: 'var(--red-muted)' }}>
                  <p className="text-[10px] mb-0.5" style={{ color: 'var(--text-muted)' }}>Stop Level</p>
                  <p className="text-sm font-mono font-semibold" style={{ color: 'var(--red)' }}>${detail.stopLevel.toFixed(2)}</p>
                  <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>${stopDist.toFixed(2)} away</p>
                </div>
                <div className="rounded-lg p-3" style={{ background: 'var(--green-muted)' }}>
                  <p className="text-[10px] mb-0.5" style={{ color: 'var(--text-muted)' }}>Target Level</p>
                  <p className="text-sm font-mono font-semibold" style={{ color: 'var(--green)' }}>${detail.targetLevel.toFixed(2)}</p>
                  <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>${targetDist.toFixed(2)} away</p>
                </div>
              </div>
              {/* R:R visual bar */}
              <div className="flex items-center gap-2">
                <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Risk/Reward</span>
                <div className="flex-1 h-2 rounded-full overflow-hidden flex" style={{ background: 'var(--bg-input)' }}>
                  <div className="h-full" style={{ width: `${(stopDist / (stopDist + targetDist)) * 100}%`, background: 'var(--red)', opacity: 0.7 }} />
                  <div className="h-full" style={{ width: `${(targetDist / (stopDist + targetDist)) * 100}%`, background: 'var(--green)', opacity: 0.7 }} />
                </div>
                <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>1:{(targetDist / stopDist).toFixed(1)}</span>
              </div>
            </ModalSection>

            <ModalSection title="Duration">
              <div className="grid grid-cols-2 gap-x-6">
                <StatRow label="Time in Trade" value={detail.timeInTrade} />
                <StatRow label="Avg Duration (BT)" value={detail.avgDuration} />
              </div>
              {detail.timeInTrade > detail.avgDuration && (
                <p className="text-[10px] mt-2 px-2 py-1 rounded" style={{ background: 'var(--orange-muted)', color: 'var(--orange)' }}>
                  Position has exceeded average backtest duration
                </p>
              )}
            </ModalSection>

            <ModalSection title="Backtest Stats for This Strategy">
              <div className="grid grid-cols-2 gap-x-6">
                <StatRow label="Win Rate" value={`${detail.btWinRate}%`} />
                <StatRow label="Avg R" value={`+${detail.btAvgR}R`} color="var(--green)" />
              </div>
            </ModalSection>

            <ModalSection title="Recent Trades (Last 5)">
              <div className="space-y-0">
                {detail.recentTrades.map((trade, i) => (
                  <div key={i} className="flex items-center justify-between py-2 border-b last:border-0" style={{ borderColor: 'var(--border)' }}>
                    <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{trade.date}</span>
                    <div className="flex items-center gap-3">
                      <span
                        className="text-[10px] px-1.5 py-0.5 rounded font-medium"
                        style={{
                          background: trade.result === 'WIN' ? 'var(--green-muted)' : 'var(--red-muted)',
                          color: trade.result === 'WIN' ? 'var(--green)' : 'var(--red)',
                        }}
                      >
                        {trade.result}
                      </span>
                      <span className="text-xs font-mono" style={{ color: trade.r >= 0 ? 'var(--green)' : 'var(--red)' }}>
                        {trade.r >= 0 ? '+' : ''}{trade.r.toFixed(1)}R
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </ModalSection>

            <ModalSection title="Close Early">
              <p className="text-[11px] mb-3" style={{ color: 'var(--text-muted)' }}>
                Closing this position early would lock in {pos.unrealizedR >= 0 ? 'a gain' : 'a loss'} of{' '}
                <span className="font-mono font-medium" style={{ color: pos.unrealizedR >= 0 ? 'var(--green)' : 'var(--red)' }}>
                  {pos.unrealizedR >= 0 ? '+' : ''}{pos.unrealizedR.toFixed(2)}R
                </span>.
                This deviates from the strategy&apos;s exit rules and may impact performance tracking.
              </p>
              <button
                className="px-4 py-2 rounded-lg text-xs font-medium transition-opacity hover:opacity-80"
                style={{ background: 'var(--red)', color: '#fff' }}
              >
                Confirm Close Early
              </button>
            </ModalSection>

            <div className="mt-3">
              <Link href={`/strategies/${detail.strategyId}`} className="text-xs font-medium" style={{ color: 'var(--accent)' }}>
                View strategy detail page &rarr;
              </Link>
            </div>
          </Modal>
        );
      })()}

      {/* 5. Health Detail Modal (per item) */}
      {showHealthDetail && (() => {
        const stratItem = MOCK_HEALTH_STRATEGIES.find(s => s.id === showHealthDetail);
        const portItem = MOCK_HEALTH_PORTFOLIOS.find(p => p.id === showHealthDetail);
        const item = stratItem || portItem;
        const detail = MOCK_HEALTH_DETAILS[showHealthDetail];
        if (!item || !detail) return null;

        const isPortfolio = !!portItem;

        return (
          <Modal
            title={`Health Detail \u2014 ${item.name}`}
            isOpen={true}
            onClose={() => setShowHealthDetail(null)}
            width="720px"
          >
            <ModalSection title="SD Deviation Explanation">
              <div className="flex items-center gap-3 mb-3">
                <span
                  className="text-2xl font-bold font-mono"
                  style={{ color: sdColor(item.deviationSd) }}
                >
                  {item.deviationSd >= 0 ? '+' : ''}{item.deviationSd.toFixed(2)} SD
                </span>
                <span
                  className="text-xs px-2 py-1 rounded-full font-medium"
                  style={{ background: sdBg(item.deviationSd), color: sdColor(item.deviationSd) }}
                >
                  {sdLabel(item.deviationSd)}
                </span>
              </div>
              <p className="text-[11px] leading-relaxed" style={{ color: 'var(--text-muted)' }}>
                {detail.explanation}
              </p>
            </ModalSection>

            <ModalSection title="Expected vs Actual Cumulative R">
              <div className="flex items-center gap-6 mb-2">
                <div className="flex items-center gap-1.5">
                  <span className="w-3 h-0.5 rounded" style={{ background: 'var(--text-muted)' }} />
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Expected</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-3 h-0.5 rounded" style={{ background: sdColor(item.deviationSd) }} />
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Actual</span>
                </div>
              </div>
              <div className="rounded-lg p-2" style={{ background: 'var(--bg-input)' }}>
                {/* Two-line sparkline comparison */}
                <svg width="100%" height="80" viewBox="0 0 300 80" preserveAspectRatio="xMidYMid meet">
                  {(() => {
                    const allVals = [...detail.expectedRHistory, ...detail.actualRHistory];
                    const min = Math.min(...allVals);
                    const max = Math.max(...allVals);
                    const range = max - min || 1;
                    const pad = 8;
                    const w = 300 - pad * 2;
                    const h = 80 - pad * 2;

                    const toPoints = (data: number[]) =>
                      data.map((v, i) => `${pad + (i / (data.length - 1)) * w},${pad + h - ((v - min) / range) * h}`).join(' ');

                    return (
                      <>
                        <polyline points={toPoints(detail.expectedRHistory)} fill="none" stroke="var(--text-muted)" strokeWidth="1.5" strokeDasharray="4 3" strokeLinejoin="round" />
                        <polyline points={toPoints(detail.actualRHistory)} fill="none" stroke={sdColor(item.deviationSd)} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
                      </>
                    );
                  })()}
                </svg>
              </div>
              <div className="flex items-center justify-between mt-2">
                <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>
                  Expected: {isPortfolio ? `$${(portItem as typeof MOCK_HEALTH_PORTFOLIOS[0]).expectedPnL.toLocaleString()}` : `${(stratItem as typeof MOCK_HEALTH_STRATEGIES[0]).expectedR}R`}
                </span>
                <span className="text-[10px] font-mono" style={{ color: sdColor(item.deviationSd) }}>
                  Actual: {isPortfolio ? `$${(portItem as typeof MOCK_HEALTH_PORTFOLIOS[0]).actualPnL.toLocaleString()}` : `${(stratItem as typeof MOCK_HEALTH_STRATEGIES[0]).actualR}R`}
                </span>
              </div>
            </ModalSection>

            <ModalSection title="Recommendation">
              <p className="text-xs leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                {detail.recommendation}
              </p>
            </ModalSection>

            <ModalSection title="Correlations">
              <div className="space-y-1.5">
                {detail.correlations.map((corr, i) => (
                  <div key={i} className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: 'var(--accent)' }} />
                    <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>{corr}</span>
                  </div>
                ))}
              </div>
            </ModalSection>

            <ModalSection title="Historical Health">
              <div className="flex items-center gap-2">
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Last week:</span>
                <span
                  className="text-[10px] px-2 py-0.5 rounded-full font-medium"
                  style={{
                    background: detail.healthyLastWeek ? 'var(--green-muted)' : 'var(--red-muted)',
                    color: detail.healthyLastWeek ? 'var(--green)' : 'var(--red)',
                  }}
                >
                  {detail.healthyLastWeek ? 'Healthy' : 'Unhealthy'}
                </span>
              </div>
            </ModalSection>

            <div className="mt-3">
              <Link
                href={isPortfolio ? `/portfolios/${item.id}` : `/strategies/${item.id}`}
                className="text-xs font-medium"
                style={{ color: 'var(--accent)' }}
              >
                View {isPortfolio ? 'portfolio' : 'strategy'} detail page &rarr;
              </Link>
            </div>
          </Modal>
        );
      })()}

      {/* 6. Market Analysis Modal */}
      <Modal
        title="Market Analysis"
        isOpen={showMarketDetail}
        onClose={() => setShowMarketDetail(false)}
        width="720px"
      >
        <ModalSection title="VIX History (Last 30 Days)">
          <div className="rounded-lg p-2" style={{ background: 'var(--bg-input)' }}>
            <MiniSparkline
              data={MOCK_VIX_HISTORY.map(v => v.value)}
              width={660}
              height={80}
              color={MOCK_MARKET.vix < 20 ? 'var(--green)' : 'var(--red)'}
            />
          </div>
          <div className="flex items-center justify-between mt-2">
            <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
              {MOCK_VIX_HISTORY[0].day}
            </span>
            <span className="text-[10px] font-mono" style={{ color: MOCK_MARKET.vix < 20 ? 'var(--green)' : 'var(--red)' }}>
              Current: {MOCK_MARKET.vix}
            </span>
            <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
              {MOCK_VIX_HISTORY[MOCK_VIX_HISTORY.length - 1].day}
            </span>
          </div>
        </ModalSection>

        <ModalSection title="Market Breadth">
          <div className="grid grid-cols-3 gap-3">
            <div className="text-center">
              <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Advancers</p>
              <p className="text-lg font-semibold" style={{ color: 'var(--green)' }}>{MOCK_MARKET.breadth.advancers}</p>
            </div>
            <div className="text-center">
              <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Decliners</p>
              <p className="text-lg font-semibold" style={{ color: 'var(--red)' }}>{MOCK_MARKET.breadth.decliners}</p>
            </div>
            <div className="text-center">
              <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>A/D Ratio</p>
              <p className="text-lg font-semibold" style={{ color: MOCK_MARKET.breadth.ratio > 1 ? 'var(--green)' : 'var(--red)' }}>
                {MOCK_MARKET.breadth.ratio.toFixed(2)}
              </p>
            </div>
          </div>
        </ModalSection>

        <ModalSection title="Index Performance">
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)' }}>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold">SPY</span>
                <span className="text-xs font-mono" style={{ color: MOCK_MARKET.spy.changePct >= 0 ? 'var(--green)' : 'var(--red)' }}>
                  {MOCK_MARKET.spy.changePct >= 0 ? '+' : ''}{MOCK_MARKET.spy.changePct.toFixed(2)}%
                </span>
              </div>
              <MiniSparkline
                data={[570, 571, 572, 573, 574, 573, 575, 576, 576.42]}
                width={280}
                height={40}
                color={MOCK_MARKET.spy.changePct >= 0 ? 'var(--green)' : 'var(--red)'}
              />
              <p className="text-xs font-mono mt-1" style={{ color: 'var(--text-primary)' }}>${MOCK_MARKET.spy.price}</p>
            </div>
            <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)' }}>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold">QQQ</span>
                <span className="text-xs font-mono" style={{ color: MOCK_MARKET.qqq.changePct >= 0 ? 'var(--green)' : 'var(--red)' }}>
                  {MOCK_MARKET.qqq.changePct >= 0 ? '+' : ''}{MOCK_MARKET.qqq.changePct.toFixed(2)}%
                </span>
              </div>
              <MiniSparkline
                data={[482, 483, 484, 485, 484, 486, 487, 487.15]}
                width={280}
                height={40}
                color={MOCK_MARKET.qqq.changePct >= 0 ? 'var(--green)' : 'var(--red)'}
              />
              <p className="text-xs font-mono mt-1" style={{ color: 'var(--text-primary)' }}>${MOCK_MARKET.qqq.price}</p>
            </div>
          </div>
        </ModalSection>

        <ModalSection title="Regime History">
          <div className="space-y-0">
            {MOCK_REGIME_HISTORY.map((entry, i) => {
              const regimeColor = entry.regime === 'Bull' ? 'var(--green)' : entry.regime === 'Bear' ? 'var(--red)' : 'var(--orange)';
              return (
                <div key={i} className="flex items-center justify-between py-2 border-b last:border-0" style={{ borderColor: 'var(--border)' }}>
                  <div className="flex items-center gap-2">
                    <span
                      className="text-[10px] px-1.5 py-0.5 rounded font-medium"
                      style={{ background: `color-mix(in srgb, ${regimeColor} 15%, transparent)`, color: regimeColor }}
                    >
                      {entry.regime}
                    </span>
                    <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                      {entry.startDate} &ndash; {entry.endDate}
                    </span>
                  </div>
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{entry.days} days</span>
                </div>
              );
            })}
          </div>
        </ModalSection>

        <ModalSection title="Impact on Your Strategies">
          <p className="text-xs leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
            Your strategies perform <span className="font-semibold" style={{ color: 'var(--green)' }}>23% better</span> in Bull regimes based on backtest data.
            Current regime has been Bull since Mar 15 (7 days). Average Bull regime duration in your backtest period: 18 days.
          </p>
        </ModalSection>
      </Modal>

      {/* 7. Goal Details Modal */}
      <Modal
        title="Goal Details \u2014 March 2026"
        isOpen={showGoalDetail}
        onClose={() => setShowGoalDetail(false)}
        width="640px"
      >
        <ModalSection title="Progress Breakdown by Week">
          {MOCK_GOAL_WEEKLY_PROGRESS.map((week, i) => (
            <div key={i} className="flex items-center justify-between py-2 border-b last:border-0" style={{ borderColor: 'var(--border)' }}>
              <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>{week.week}</span>
              <div className="flex items-center gap-3">
                <span className="text-xs font-mono" style={{ color: week.earned >= week.target ? 'var(--green)' : 'var(--text-muted)' }}>
                  ${week.earned.toLocaleString()} / ${week.target.toLocaleString()}
                </span>
                <span
                  className="text-[10px] px-1.5 py-0.5 rounded font-medium"
                  style={{
                    background: week.pace === 'ahead' ? 'var(--green-muted)' : 'var(--orange-muted)',
                    color: week.pace === 'ahead' ? 'var(--green)' : 'var(--orange)',
                  }}
                >
                  {week.pace === 'ahead' ? 'Ahead' : 'Behind'}
                </span>
              </div>
            </div>
          ))}
        </ModalSection>

        <ModalSection title="Pace Analysis">
          {(() => {
            const daysRemaining = 9;
            const remaining = MOCK_PORTFOLIO.monthlyTarget - MOCK_PORTFOLIO.monthlyProgress;
            const dailyRate = Math.round(remaining / daysRemaining);
            const behind = remaining > 0;
            return (
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <span
                    className="text-xs px-2 py-1 rounded font-medium"
                    style={{
                      background: behind ? 'var(--orange-muted)' : 'var(--green-muted)',
                      color: behind ? 'var(--orange)' : 'var(--green)',
                    }}
                  >
                    {behind ? `Behind by $${remaining.toLocaleString()}` : 'On track'}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-x-6">
                  <StatRow label="Days Remaining" value={String(daysRemaining)} />
                  <StatRow label="Required Daily Rate" value={`$${dailyRate}/day`} color={dailyRate > 300 ? 'var(--orange)' : 'var(--green)'} />
                </div>
              </div>
            );
          })()}
        </ModalSection>

        <ModalSection title="Best Contributing Strategy">
          <div className="flex items-center gap-3">
            <div className="flex-1">
              <p className="text-xs font-medium" style={{ color: 'var(--text-primary)' }}>AAPL LONG - Mass #5</p>
              <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>31 trades, +24.8R cumulative</p>
            </div>
            <span className="text-sm font-mono font-semibold" style={{ color: 'var(--green)' }}>+$1,240</span>
          </div>
        </ModalSection>

        <ModalSection title="Historical Goal Achievement">
          <div className="space-y-0">
            {MOCK_GOAL_HISTORY.map((month, i) => (
              <div key={i} className="flex items-center justify-between py-2 border-b last:border-0" style={{ borderColor: 'var(--border)' }}>
                <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>{month.month}</span>
                <div className="flex items-center gap-3">
                  <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
                    ${month.achieved.toLocaleString()} / ${month.target.toLocaleString()}
                  </span>
                  <span
                    className="text-[10px] px-1.5 py-0.5 rounded font-medium"
                    style={{
                      background: month.hit ? 'var(--green-muted)' : 'var(--red-muted)',
                      color: month.hit ? 'var(--green)' : 'var(--red)',
                    }}
                  >
                    {month.hit ? 'Hit' : 'Missed'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </ModalSection>
      </Modal>

      {/* 8. Issues & Warnings Detail Modal */}
      <Modal
        title="Issues & Warnings"
        isOpen={showIssuesDetail}
        onClose={() => setShowIssuesDetail(false)}
        width="720px"
      >
        <ModalSection title={`Active Issues (${MOCK_ISSUES.length})`}>
          <div className="space-y-2">
            {MOCK_ISSUES.map((issue) => {
              const style = severityStyles[issue.severity] || severityStyles.info;
              return (
                <div
                  key={issue.id}
                  className="rounded-lg p-3"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
                >
                  <div className="flex items-start gap-2.5 mb-2">
                    <div
                      className="w-5 h-5 rounded flex items-center justify-center text-[10px] font-bold flex-shrink-0 mt-0.5"
                      style={{ background: style.bg, color: style.color }}
                    >
                      {style.icon}
                    </div>
                    <div className="flex-1">
                      <p className="text-xs leading-relaxed mb-1" style={{ color: 'var(--text-secondary)' }}>
                        {issue.message}
                      </p>
                      <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                        {issue.severity === 'warning' && 'Suggested: Monitor position duration and consider manual review if duration exceeds 2x average.'}
                        {issue.severity === 'info' && 'Suggested: Price drift within acceptable range. Will auto-reconcile on next bar close.'}
                        {issue.severity === 'error' && 'Suggested: Check webhook endpoint URL and authentication. Re-test from Alerts > Webhook Templates.'}
                      </p>
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      <span
                        className="text-[9px] px-1.5 py-0.5 rounded"
                        style={{ background: style.bg, color: style.color }}
                      >
                        {issue.severity}
                      </span>
                      <button
                        className="text-[9px] px-2 py-0.5 rounded transition-colors"
                        style={{ background: 'var(--bg-card)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}
                        title="Dismiss this issue"
                      >
                        Dismiss
                      </button>
                    </div>
                  </div>
                  <Link href={issue.link} className="text-[10px] font-medium" style={{ color: 'var(--accent)' }}>
                    View related page &rarr;
                  </Link>
                </div>
              );
            })}
          </div>
        </ModalSection>

        <ModalSection title={`Resolved Issues (${MOCK_RESOLVED_ISSUES.length})`}>
          <div className="space-y-0">
            {MOCK_RESOLVED_ISSUES.map((issue) => {
              const style = severityStyles[issue.severity] || severityStyles.info;
              return (
                <div key={issue.id} className="flex items-start gap-2.5 py-2.5 border-b last:border-0" style={{ borderColor: 'var(--border)' }}>
                  <div
                    className="w-5 h-5 rounded flex items-center justify-center text-[10px] font-bold flex-shrink-0 mt-0.5 opacity-50"
                    style={{ background: style.bg, color: style.color }}
                  >
                    {style.icon}
                  </div>
                  <div className="flex-1">
                    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{issue.message}</p>
                  </div>
                  <span className="text-[10px] flex-shrink-0" style={{ color: 'var(--text-muted)' }}>
                    {issue.resolvedAt}
                  </span>
                </div>
              );
            })}
          </div>
        </ModalSection>
      </Modal>

      {/* 9. Activity Log Modal */}
      <Modal
        title="Activity Log"
        isOpen={showActivityDetail}
        onClose={() => setShowActivityDetail(false)}
        width="720px"
      >
        {/* Filters */}
        <div className="flex items-center gap-3 mb-4">
          <div className="flex gap-0.5 rounded-lg overflow-hidden" style={{ border: '1px solid var(--border)' }}>
            {(['all', 'entry', 'exit', 'system', 'summary'] as const).map((filter) => (
              <button
                key={filter}
                onClick={() => setActivityFilter(filter)}
                className="px-2.5 py-1 text-[10px] font-medium capitalize"
                style={{
                  background: activityFilter === filter ? 'var(--accent-muted)' : 'transparent',
                  color: activityFilter === filter ? 'var(--accent)' : 'var(--text-muted)',
                }}
              >
                {filter}
              </button>
            ))}
          </div>
          <div className="flex-1">
            <input
              type="text"
              placeholder="Search activity..."
              value={activitySearch}
              onChange={(e) => setActivitySearch(e.target.value)}
              className="w-full px-3 py-1.5 rounded-lg text-xs"
              style={{
                background: 'var(--bg-input)',
                border: '1px solid var(--border)',
                color: 'var(--text-primary)',
                outline: 'none',
              }}
            />
          </div>
        </div>

        {/* Activity List */}
        <div className="space-y-0">
          {filteredActivity.length === 0 && (
            <p className="text-xs text-center py-6" style={{ color: 'var(--text-muted)' }}>No matching activity</p>
          )}
          {filteredActivity.map((alert, idx) => {
            const typeStyle = alertTypeStyles[alert.type] || alertTypeStyles.system;
            return (
              <div
                key={idx}
                className="flex items-center gap-2.5 py-2.5 border-b last:border-0"
                style={{ borderColor: 'var(--border)' }}
              >
                <div
                  className="w-5 h-5 rounded flex items-center justify-center text-[10px] font-bold flex-shrink-0"
                  style={{ background: typeStyle.bg, color: typeStyle.color }}
                >
                  {typeStyle.icon}
                </div>
                <p className="text-xs flex-1" style={{ color: 'var(--text-secondary)' }}>
                  {alert.message}
                </p>
                <div className="flex items-center gap-2 flex-shrink-0">
                  <span
                    className="text-[9px] px-1.5 py-0.5 rounded capitalize"
                    style={{ background: typeStyle.bg, color: typeStyle.color }}
                  >
                    {alert.type}
                  </span>
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                    {alert.time}
                  </span>
                </div>
              </div>
            );
          })}
        </div>

        <div className="mt-3 pt-3 border-t" style={{ borderColor: 'var(--border)' }}>
          <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
            Showing {filteredActivity.length} of {MOCK_FULL_ACTIVITY.length} activities
          </p>
        </div>
      </Modal>

    </div>
  );
}
