'use client';

import { useState, useMemo, useCallback, useEffect } from 'react';
import Link from 'next/link';
import Card from '@/components/Card';
import MetricCard from '@/components/MetricCard';
import Modal from '@/components/Modal';

// =============================================================================
// V4 Meta: Creative — Trading Cockpit + Portfolio Pulse
// =============================================================================
// The "wow factor" dashboard. Key creative features:
//
// 1. DAILY BRIEFING — personalized greeting with market status summary
// 2. STRATEGY LEADERBOARD — ranked strategies with trend sparklines
// 3. HEAT MAP CALENDAR — trading day P&L as colored cells
// 4. PORTFOLIO PULSE — animated heartbeat line showing real-time activity
// 5. ACTIVE POSITIONS WIDGET — live position tracker with unrealized P&L
// 6. MARKET REGIME INDICATOR — Bull/Bear/Neutral gauge with VIX + breadth
// 7. GOAL TRACKER — monthly target progress bar
// 8. NOTIFICATION CENTER — bell icon with unread count, alert feed
// 9. DENSE COCKPIT LAYOUT — Bloomberg-style grid, information-rich
// =============================================================================

// ---------------------------------------------------------------------------
// CSS Animations
// ---------------------------------------------------------------------------

const ANIMATION_STYLES = `
@keyframes heartbeat-line {
  0% { stroke-dashoffset: 800; }
  100% { stroke-dashoffset: 0; }
}
@keyframes pulse-dot {
  0%, 100% { opacity: 0.4; transform: scale(1); }
  50% { opacity: 1; transform: scale(1.5); }
}
@keyframes fade-in-up {
  0% { opacity: 0; transform: translateY(8px); }
  100% { opacity: 1; transform: translateY(0); }
}
@keyframes glow-border {
  0%, 100% { box-shadow: 0 0 6px rgba(0,255,136,0.1); }
  50% { box-shadow: 0 0 16px rgba(0,255,136,0.25); }
}
@keyframes regime-pulse {
  0%, 100% { opacity: 0.6; }
  50% { opacity: 1; }
}
@keyframes progress-fill {
  0% { width: 0%; }
}
@keyframes count-up {
  0% { opacity: 0; transform: translateY(4px); }
  100% { opacity: 1; transform: translateY(0); }
}
@keyframes slide-in-right {
  0% { transform: translateX(20px); opacity: 0; }
  100% { transform: translateX(0); opacity: 1; }
}
@keyframes bell-ring {
  0%, 100% { transform: rotate(0); }
  10% { transform: rotate(12deg); }
  20% { transform: rotate(-10deg); }
  30% { transform: rotate(6deg); }
  40% { transform: rotate(-4deg); }
  50% { transform: rotate(0); }
}
`;

// ---------------------------------------------------------------------------
// Mock Data
// ---------------------------------------------------------------------------

const MOCK_USER = { name: 'Kevin', greeting: 'Good morning' };

const MOCK_MARKET = {
  session: 'RTH',
  sessionLabel: 'Regular Trading Hours',
  timeUntilClose: '4h 28m',
  spy: { price: 576.42, change: +0.87, changePct: +0.15 },
  qqq: { price: 487.15, change: +2.34, changePct: +0.48 },
  vix: { value: 14.2, label: 'Low Volatility' },
  breadth: { advancers: 312, decliners: 188, ratio: 1.66 },
  regime: 'Bull' as const,
};

const MOCK_PORTFOLIO = {
  totalPnL: 4230,
  todayPnL: 247,
  todayPnLPct: 0.31,
  balance: 84230,
  startingBalance: 80000,
  maxDD: -2.1,
  winRate: 55.2,
  monthlyTarget: 5000,
  monthlyProgress: 3247,
};

const MOCK_STRATEGIES = [
  { id: '1', name: 'SPY LONG - Mass #1', symbol: 'SPY', direction: 'LONG', winRate: 62.5, dailyR: 2.41, totalR: 48.2, trend: [0, 1.2, 0.8, 2.4, 1.8, 3.2, 2.9, 4.1, 3.5, 5.0], rank: 1, status: 'active' as const },
  { id: '2', name: 'NVDA LONG - Momentum', symbol: 'NVDA', direction: 'LONG', winRate: 58.3, dailyR: 1.89, totalR: 35.7, trend: [0, 0.5, -0.3, 1.2, 0.8, 1.8, 2.5, 2.1, 3.0, 3.5], rank: 2, status: 'active' as const },
  { id: '3', name: 'QQQ LONG - VWAP Bounce', symbol: 'QQQ', direction: 'LONG', winRate: 55.1, dailyR: 1.24, totalR: 22.3, trend: [0, 0.3, 0.8, 0.5, 1.2, 0.9, 1.5, 1.8, 2.1, 2.5], rank: 3, status: 'active' as const },
  { id: '4', name: 'TSLA SHORT - Reversal', symbol: 'TSLA', direction: 'SHORT', winRate: 51.2, dailyR: 0.67, totalR: 8.4, trend: [0, -0.5, 0.2, -0.3, 0.8, 0.5, 1.2, 0.9, 1.5, 1.1], rank: 4, status: 'idle' as const },
  { id: '5', name: 'AAPL LONG - Trend Follow', symbol: 'AAPL', direction: 'LONG', winRate: 53.8, dailyR: 0.42, totalR: 5.8, trend: [0, 0.2, -0.1, 0.5, 0.3, 0.8, 0.6, 1.0, 0.7, 1.2], rank: 5, status: 'idle' as const },
];

const MOCK_POSITIONS = [
  { id: '1', strategy: 'SPY LONG - Mass #1', symbol: 'SPY', direction: 'LONG', entryPrice: 574.80, currentPrice: 576.42, unrealizedR: 1.35, entryTime: '09:35 AM', status: 'open' as const },
  { id: '2', strategy: 'NVDA LONG - Momentum', symbol: 'NVDA', direction: 'LONG', entryPrice: 896.50, currentPrice: 901.20, unrealizedR: 0.62, entryTime: '10:12 AM', status: 'open' as const },
];

const MOCK_ALERTS = [
  { id: '1', time: '10:32 AM', message: 'NVDA LONG entry signal fired', type: 'entry' as const, read: false },
  { id: '2', time: '10:15 AM', message: 'SPY LONG exit -- +1.2R', type: 'exit' as const, read: false },
  { id: '3', time: '09:45 AM', message: 'Monitor started for 4 strategies', type: 'system' as const, read: true },
  { id: '4', time: '09:30 AM', message: 'Market open -- RTH session started', type: 'system' as const, read: true },
  { id: '5', time: 'Yesterday', message: 'Daily summary: +$142, 3W 1L', type: 'summary' as const, read: true },
];

// Calendar data: day -> P&L for current month
const CALENDAR_DATA: Record<number, number> = {
  1: 142, 2: -67, 3: 210, 4: 55, 5: -120,
  6: 0, 7: 0, // weekend
  8: 180, 9: 95, 10: -42, 11: 310, 12: -85,
  13: 0, 14: 0, // weekend
  15: 165, 16: 220, 17: -30, 18: 145, 19: 280,
  20: 0, 21: 0, // weekend -- today is 21
};

// Portfolio pulse (heartbeat data — R values over time)
const PULSE_DATA = [
  0, 0.2, 0.5, 0.3, 0.8, 1.2, 0.9, 1.5, 1.8, 1.4, 2.0, 2.5, 2.2, 2.8, 3.1, 2.7,
  3.4, 3.8, 3.5, 4.1, 4.5, 4.2, 4.8, 5.2, 4.9, 5.5, 5.8, 5.4, 6.0, 6.4,
];

// ---------------------------------------------------------------------------
// SVG Components
// ---------------------------------------------------------------------------

/** Mini trend sparkline for strategy leaderboard */
function TrendSparkline({
  data,
  width = 80,
  height = 24,
  color,
}: {
  data: number[];
  width?: number;
  height?: number;
  color: string;
}) {
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((v - min) / range) * (height - 4) - 2;
    return `${x},${y}`;
  });
  const pathD = `M${points.join(' L')}`;

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <path d={pathD} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

/** Portfolio Pulse — heartbeat-style equity line */
function PortfolioPulse({
  data,
  width = 600,
  height = 80,
}: {
  data: number[];
  width?: number;
  height?: number;
}) {
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const pad = 4;
  const chartW = width - pad * 2;
  const chartH = height - pad * 2;

  const points = data.map((v, i) => {
    const x = pad + (i / (data.length - 1)) * chartW;
    const y = pad + chartH - ((v - min) / range) * chartH;
    return `${x},${y}`;
  });

  const pathD = `M${points.join(' L')}`;
  const fillD = `${pathD} L${pad + chartW},${pad + chartH} L${pad},${pad + chartH} Z`;
  const lastPt = points[points.length - 1].split(',');
  const color = data[data.length - 1] >= 0 ? 'var(--green)' : 'var(--red)';

  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet">
      <defs>
        <linearGradient id="pulseGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.15" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={fillD} fill="url(#pulseGrad)" />
      <path
        d={pathD}
        fill="none"
        stroke={color}
        strokeWidth="2"
        strokeLinejoin="round"
        strokeDasharray="800"
        style={{ animation: 'heartbeat-line 2s ease-out forwards' }}
      />
      {/* Live dot at end */}
      <circle
        cx={lastPt[0]}
        cy={lastPt[1]}
        r="4"
        fill={color}
        style={{ animation: 'pulse-dot 2s ease-in-out infinite' }}
      />
      <circle cx={lastPt[0]} cy={lastPt[1]} r="7" fill={color} opacity="0.2" />
    </svg>
  );
}

/** Market Regime Gauge */
function RegimeGauge({
  regime,
  size = 120,
}: {
  regime: 'Bull' | 'Bear' | 'Neutral';
  size?: number;
}) {
  const c = size / 2;
  const r = (size - 16) / 2;
  const circumference = 2 * Math.PI * r;
  // Three-quarter arc
  const arcLen = circumference * 0.75;

  const regimeData = {
    Bull: { pct: 0.78, color: 'var(--green)', label: 'BULLISH' },
    Bear: { pct: 0.25, color: 'var(--red)', label: 'BEARISH' },
    Neutral: { pct: 0.50, color: 'var(--orange)', label: 'NEUTRAL' },
  };

  const { pct, color, label } = regimeData[regime];
  const fillLen = arcLen * pct;

  return (
    <div className="relative flex flex-col items-center">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {/* Background arc */}
        <circle
          cx={c} cy={c} r={r} fill="none"
          stroke="var(--bg-input)" strokeWidth="8"
          strokeDasharray={`${arcLen} ${circumference}`}
          strokeLinecap="round"
          transform={`rotate(135 ${c} ${c})`}
        />
        {/* Colored arc */}
        <circle
          cx={c} cy={c} r={r} fill="none"
          stroke={color} strokeWidth="8"
          strokeDasharray={`${fillLen} ${circumference}`}
          strokeLinecap="round"
          transform={`rotate(135 ${c} ${c})`}
          style={{ animation: 'regime-pulse 3s ease-in-out infinite' }}
        />
        {/* Glow */}
        <circle
          cx={c} cy={c} r={r} fill="none"
          stroke={color} strokeWidth="14"
          strokeDasharray={`${fillLen} ${circumference}`}
          strokeLinecap="round"
          transform={`rotate(135 ${c} ${c})`}
          opacity="0.15"
          style={{ filter: 'blur(4px)' }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-lg font-bold" style={{ color }}>{label}</span>
        <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Market Regime</span>
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
  // March 2026 starts on Sunday — generate a 5-week grid
  // Simplified: just show days 1-21 in a 3-row grid (weeks)
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

  return (
    <div>
      {/* Day headers */}
      <div className="grid grid-cols-7 gap-1 mb-1">
        {dayNames.map(d => (
          <div key={d} className="text-center text-[9px]" style={{ color: 'var(--text-muted)' }}>{d}</div>
        ))}
      </div>
      {/* Weeks */}
      {weeks.map((week, wi) => (
        <div key={wi} className="grid grid-cols-7 gap-1 mb-1">
          {week.map(day => {
            const val = data[day];
            const isToday = day === currentDay;
            const isWeekend = day % 7 === 6 || day % 7 === 0;
            return (
              <div
                key={day}
                className="aspect-square rounded-md flex flex-col items-center justify-center relative"
                style={{
                  background: isWeekend ? 'var(--bg-input)' : getColor(day),
                  border: isToday ? '2px solid var(--accent)' : '1px solid transparent',
                  opacity: isWeekend ? 0.3 : 1,
                }}
                title={val !== undefined ? `Day ${day}: ${val >= 0 ? '+' : ''}$${val}` : `Day ${day}`}
              >
                <span className="text-[9px] font-medium" style={{ color: 'var(--text-secondary)' }}>{day}</span>
                {val !== undefined && val !== 0 && !isWeekend && (
                  <span className="text-[8px] font-mono" style={{ color: val > 0 ? 'var(--green)' : 'var(--red)' }}>
                    {val > 0 ? '+' : ''}{val}
                  </span>
                )}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}

/** Goal Progress Bar */
function GoalProgress({
  current,
  target,
  label,
}: {
  current: number;
  target: number;
  label: string;
}) {
  const pct = Math.min((current / target) * 100, 100);
  const onTrack = pct >= (new Date().getDate() / 30) * 100;

  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>{label}</span>
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
            animation: 'progress-fill 1s ease-out',
            boxShadow: `0 0 8px ${onTrack ? 'rgba(76,175,80,0.4)' : 'rgba(255,152,0,0.4)'}`,
          }}
        />
        {/* Track line (expected progress) */}
        <div
          className="absolute top-0 h-full w-0.5"
          style={{
            left: `${(new Date().getDate() / 30) * 100}%`,
            background: 'var(--text-muted)',
            opacity: 0.5,
          }}
        />
      </div>
      <div className="flex items-center justify-between mt-1">
        <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
          ${current.toLocaleString()} / ${target.toLocaleString()}
        </span>
        <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
          {onTrack ? 'On track' : 'Behind pace'}
        </span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function DashboardV4() {
  const [showNotifications, setShowNotifications] = useState(false);
  const unreadCount = MOCK_ALERTS.filter(a => !a.read).length;

  const alertTypeColors: Record<string, { bg: string; color: string; icon: string }> = {
    entry: { bg: 'var(--green-muted)', color: 'var(--green)', icon: '\u2191' },
    exit: { bg: 'var(--red-muted)', color: 'var(--red)', icon: '\u2193' },
    system: { bg: 'rgba(156,163,175,0.15)', color: 'var(--text-muted)', icon: 'S' },
    summary: { bg: 'var(--accent-muted)', color: 'var(--accent)', icon: '\u2211' },
  };

  useEffect(() => {
    const id = 'v4-animation-styles';
    if (!document.getElementById(id)) {
      const style = document.createElement('style');
      style.id = id;
      style.textContent = ANIMATION_STYLES;
      document.head.appendChild(style);
    }
  }, []);

  return (
    <div style={{ animation: 'fade-in-up 0.3s ease-out' }}>

      {/* ============ HEADER: Briefing + Notifications ============ */}
      <div className="flex items-start justify-between mb-5">
        <div>
          <h1 className="text-2xl font-bold mb-1">{MOCK_USER.greeting}, {MOCK_USER.name}</h1>
          <div className="flex items-center gap-3">
            <span className="text-sm" style={{ color: 'var(--text-muted)' }}>
              Markets are open
            </span>
            <span
              className="flex items-center gap-1.5 text-xs px-2.5 py-0.5 rounded-full"
              style={{ background: 'var(--green-muted)', color: 'var(--green)' }}
            >
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--green)' }} />
              {MOCK_MARKET.session}
            </span>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
              Closes in {MOCK_MARKET.timeUntilClose}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {/* Market tickers */}
          <div className="flex gap-2 mr-2">
            {[
              { label: 'SPY', ...MOCK_MARKET.spy },
              { label: 'QQQ', ...MOCK_MARKET.qqq },
            ].map(ticker => (
              <div
                key={ticker.label}
                className="px-2.5 py-1.5 rounded-lg text-xs"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
              >
                <span className="font-medium mr-1.5">{ticker.label}</span>
                <span className="font-mono">{ticker.price}</span>
                <span
                  className="ml-1 font-mono"
                  style={{ color: ticker.changePct >= 0 ? 'var(--green)' : 'var(--red)' }}
                >
                  {ticker.changePct >= 0 ? '+' : ''}{ticker.changePct.toFixed(2)}%
                </span>
              </div>
            ))}
          </div>

          {/* Notification bell */}
          <button
            className="relative w-9 h-9 rounded-lg flex items-center justify-center"
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
            onClick={() => setShowNotifications(true)}
          >
            <span
              style={{
                fontSize: '16px',
                animation: unreadCount > 0 ? 'bell-ring 2s ease-in-out infinite' : undefined,
                display: 'inline-block',
              }}
            >
              {'\u{1F514}'}
            </span>
            {unreadCount > 0 && (
              <span
                className="absolute -top-1 -right-1 w-4 h-4 rounded-full flex items-center justify-center text-[9px] font-bold"
                style={{ background: 'var(--red)', color: 'white' }}
              >
                {unreadCount}
              </span>
            )}
          </button>
        </div>
      </div>

      {/* ============ DAILY BRIEFING CARD ============ */}
      <Card className="mb-4">
        <div className="flex items-center gap-6">
          <div className="flex-1">
            <p className="text-sm leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
              Your best strategy is <strong style={{ color: 'var(--accent)' }}>SPY LONG - Mass #1</strong> at{' '}
              <span style={{ color: 'var(--green)' }}>+2.41R/day</span>.{' '}
              {MOCK_ALERTS.filter(a => a.type === 'entry' || a.type === 'exit').length} alerts fired today.{' '}
              Portfolio is{' '}
              <span style={{ color: MOCK_PORTFOLIO.todayPnL >= 0 ? 'var(--green)' : 'var(--red)' }}>
                {MOCK_PORTFOLIO.todayPnL >= 0 ? 'up' : 'down'} ${Math.abs(MOCK_PORTFOLIO.todayPnL)}
              </span>{' '}
              ({MOCK_PORTFOLIO.todayPnLPct >= 0 ? '+' : ''}{MOCK_PORTFOLIO.todayPnLPct.toFixed(2)}%).{' '}
              {MOCK_POSITIONS.length} position{MOCK_POSITIONS.length !== 1 ? 's' : ''} open.
            </p>
          </div>
          <div className="flex gap-2">
            <Link href="/strategy-builder">
              <button
                className="px-3 py-1.5 rounded-lg text-xs font-medium"
                style={{ background: 'var(--accent)', color: '#000' }}
              >
                Build Strategy
              </button>
            </Link>
            <Link href="/alerts">
              <button
                className="px-3 py-1.5 rounded-lg text-xs font-medium"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
              >
                View Alerts
              </button>
            </Link>
          </div>
        </div>
      </Card>

      {/* ============ TOP KPI ROW ============ */}
      <div className="grid grid-cols-6 gap-3 mb-4">
        <MetricCard
          label="Today P&L"
          value={`+$${MOCK_PORTFOLIO.todayPnL}`}
          delta={`+${MOCK_PORTFOLIO.todayPnLPct.toFixed(2)}%`}
          positive
        />
        <MetricCard
          label="Portfolio Value"
          value={`$${MOCK_PORTFOLIO.balance.toLocaleString()}`}
          delta={`+$${MOCK_PORTFOLIO.totalPnL.toLocaleString()}`}
          positive
        />
        <MetricCard
          label="Active Strategies"
          value={String(MOCK_STRATEGIES.filter(s => s.status === 'active').length)}
          delta={`${MOCK_STRATEGIES.length} total`}
        />
        <MetricCard
          label="Open Positions"
          value={String(MOCK_POSITIONS.length)}
          delta="Monitored"
        />
        <MetricCard
          label="Win Rate"
          value={`${MOCK_PORTFOLIO.winRate}%`}
          delta="Portfolio-wide"
        />
        <MetricCard
          label="VIX"
          value={MOCK_MARKET.vix.value.toFixed(1)}
          delta={MOCK_MARKET.vix.label}
          positive={MOCK_MARKET.vix.value < 20}
        />
      </div>

      {/* ============ MAIN GRID: Dense Cockpit Layout ============ */}
      <div className="grid grid-cols-12 gap-4">
        {/* -- LEFT COLUMN: Strategies + Positions (5 cols) -- */}
        <div className="col-span-5 space-y-4">
          {/* Strategy Leaderboard */}
          <Card>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>Strategy Leaderboard</h3>
              <Link href="/strategies">
                <span className="text-[10px]" style={{ color: 'var(--accent)', cursor: 'pointer' }}>View All</span>
              </Link>
            </div>
            <div className="space-y-0">
              {MOCK_STRATEGIES.map((strategy, idx) => (
                <Link key={strategy.id} href={`/strategies/${strategy.id}`}>
                  <div
                    className="flex items-center gap-3 py-2.5 border-b last:border-0 transition-colors"
                    style={{
                      borderColor: 'var(--border)',
                      animation: `slide-in-right 0.3s ease-out ${idx * 0.05}s both`,
                    }}
                  >
                    {/* Rank */}
                    <span
                      className="w-6 h-6 rounded-md flex items-center justify-center text-[10px] font-bold flex-shrink-0"
                      style={{
                        background: strategy.rank <= 3 ? 'var(--accent-muted)' : 'var(--bg-input)',
                        color: strategy.rank <= 3 ? 'var(--accent)' : 'var(--text-muted)',
                      }}
                    >
                      #{strategy.rank}
                    </span>

                    {/* Info */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5">
                        <p className="text-xs font-medium truncate">{strategy.name}</p>
                        {strategy.status === 'active' && (
                          <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: 'var(--green)' }} />
                        )}
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                          WR {strategy.winRate}%
                        </span>
                        <span className="text-[10px] font-mono" style={{ color: 'var(--green)' }}>
                          +{strategy.dailyR.toFixed(2)}R/d
                        </span>
                      </div>
                    </div>

                    {/* Trend sparkline */}
                    <TrendSparkline
                      data={strategy.trend}
                      width={60}
                      height={20}
                      color={strategy.trend[strategy.trend.length - 1] >= strategy.trend[0] ? 'var(--green)' : 'var(--red)'}
                    />

                    {/* Total R */}
                    <span
                      className="text-xs font-mono font-semibold w-12 text-right flex-shrink-0"
                      style={{ color: strategy.totalR >= 0 ? 'var(--green)' : 'var(--red)' }}
                    >
                      {strategy.totalR >= 0 ? '+' : ''}{strategy.totalR.toFixed(1)}R
                    </span>
                  </div>
                </Link>
              ))}
            </div>
          </Card>

          {/* Active Positions */}
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
            {MOCK_POSITIONS.map(pos => (
              <div
                key={pos.id}
                className="rounded-lg p-3 mb-2 last:mb-0"
                style={{
                  background: 'var(--bg-input)',
                  border: '1px solid var(--border)',
                  animation: 'glow-border 3s ease-in-out infinite',
                }}
              >
                <div className="flex items-center justify-between mb-2">
                  <div>
                    <span className="text-sm font-semibold">{pos.symbol}</span>
                    <span
                      className="ml-1.5 text-[10px] px-1.5 py-0.5 rounded"
                      style={{
                        background: pos.direction === 'LONG' ? 'var(--green-muted)' : 'var(--red-muted)',
                        color: pos.direction === 'LONG' ? 'var(--green)' : 'var(--red)',
                      }}
                    >
                      {pos.direction}
                    </span>
                  </div>
                  <span
                    className="text-sm font-mono font-bold"
                    style={{ color: pos.unrealizedR >= 0 ? 'var(--green)' : 'var(--red)' }}
                  >
                    {pos.unrealizedR >= 0 ? '+' : ''}{pos.unrealizedR.toFixed(2)}R
                  </span>
                </div>
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
        </div>

        {/* -- CENTER COLUMN: Pulse + Calendar (4 cols) -- */}
        <div className="col-span-4 space-y-4">
          {/* Portfolio Pulse */}
          <Card>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>Portfolio Pulse</h3>
              <span className="text-[10px] font-mono" style={{ color: 'var(--green)' }}>
                +{PULSE_DATA[PULSE_DATA.length - 1].toFixed(1)}R total
              </span>
            </div>
            <PortfolioPulse data={PULSE_DATA} width={480} height={100} />
          </Card>

          {/* P&L Calendar */}
          <Card>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>March 2026 P&L Calendar</h3>
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
            </div>
            <PnLCalendar data={CALENDAR_DATA} currentDay={21} />
            <div className="flex justify-between mt-2 pt-2 border-t" style={{ borderColor: 'var(--border)' }}>
              <div>
                <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Win Days</span>
                <p className="text-sm font-semibold" style={{ color: 'var(--green)' }}>
                  {Object.values(CALENDAR_DATA).filter(v => v > 0).length}
                </p>
              </div>
              <div>
                <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Loss Days</span>
                <p className="text-sm font-semibold" style={{ color: 'var(--red)' }}>
                  {Object.values(CALENDAR_DATA).filter(v => v < 0).length}
                </p>
              </div>
              <div>
                <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Best Day</span>
                <p className="text-sm font-semibold" style={{ color: 'var(--green)' }}>
                  +${Math.max(...Object.values(CALENDAR_DATA))}
                </p>
              </div>
              <div>
                <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Worst Day</span>
                <p className="text-sm font-semibold" style={{ color: 'var(--red)' }}>
                  ${Math.min(...Object.values(CALENDAR_DATA).filter(v => v !== 0))}
                </p>
              </div>
            </div>
          </Card>

          {/* Recent Activity */}
          <Card>
            <h3 className="text-xs font-semibold mb-3" style={{ color: 'var(--text-muted)' }}>Recent Activity</h3>
            <div className="space-y-0">
              {MOCK_ALERTS.map((alert) => {
                const typeStyle = alertTypeColors[alert.type] || alertTypeColors.system;
                return (
                  <div
                    key={alert.id}
                    className="flex items-center gap-2.5 py-2 border-b last:border-0"
                    style={{ borderColor: 'var(--border)' }}
                  >
                    <div
                      className="w-6 h-6 rounded-md flex items-center justify-center text-[10px] font-bold flex-shrink-0"
                      style={{ background: typeStyle.bg, color: typeStyle.color }}
                    >
                      {typeStyle.icon}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-xs truncate" style={{ color: 'var(--text-secondary)' }}>{alert.message}</p>
                    </div>
                    <span className="text-[10px] flex-shrink-0" style={{ color: 'var(--text-muted)' }}>{alert.time}</span>
                    {!alert.read && (
                      <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: 'var(--accent)' }} />
                    )}
                  </div>
                );
              })}
            </div>
          </Card>
        </div>

        {/* -- RIGHT COLUMN: Market + Goal + System (3 cols) -- */}
        <div className="col-span-3 space-y-4">
          {/* Market Regime */}
          <Card>
            <h3 className="text-xs font-semibold mb-2" style={{ color: 'var(--text-muted)' }}>Market Regime</h3>
            <div className="flex justify-center mb-2">
              <RegimeGauge regime={MOCK_MARKET.regime} size={110} />
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span style={{ color: 'var(--text-muted)' }}>VIX</span>
                <span className="font-mono" style={{ color: MOCK_MARKET.vix.value < 20 ? 'var(--green)' : 'var(--red)' }}>
                  {MOCK_MARKET.vix.value.toFixed(1)}
                </span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span style={{ color: 'var(--text-muted)' }}>Market Breadth</span>
                <span className="font-mono" style={{ color: MOCK_MARKET.breadth.ratio > 1 ? 'var(--green)' : 'var(--red)' }}>
                  {MOCK_MARKET.breadth.ratio.toFixed(2)}
                </span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span style={{ color: 'var(--text-muted)' }}>Advancers</span>
                <span className="font-mono" style={{ color: 'var(--green)' }}>{MOCK_MARKET.breadth.advancers}</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span style={{ color: 'var(--text-muted)' }}>Decliners</span>
                <span className="font-mono" style={{ color: 'var(--red)' }}>{MOCK_MARKET.breadth.decliners}</span>
              </div>
            </div>
          </Card>

          {/* Monthly Goal */}
          <Card>
            <h3 className="text-xs font-semibold mb-3" style={{ color: 'var(--text-muted)' }}>Monthly Goal</h3>
            <GoalProgress
              current={MOCK_PORTFOLIO.monthlyProgress}
              target={MOCK_PORTFOLIO.monthlyTarget}
              label="March Target"
            />
            <div className="grid grid-cols-2 gap-2 mt-3">
              <div className="px-2 py-1.5 rounded-lg text-center" style={{ background: 'var(--bg-input)' }}>
                <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Remaining</span>
                <p className="text-sm font-mono font-semibold">
                  ${(MOCK_PORTFOLIO.monthlyTarget - MOCK_PORTFOLIO.monthlyProgress).toLocaleString()}
                </p>
              </div>
              <div className="px-2 py-1.5 rounded-lg text-center" style={{ background: 'var(--bg-input)' }}>
                <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Daily Needed</span>
                <p className="text-sm font-mono font-semibold">
                  ${Math.round((MOCK_PORTFOLIO.monthlyTarget - MOCK_PORTFOLIO.monthlyProgress) / (30 - 21))}
                </p>
              </div>
            </div>
          </Card>

          {/* System Status */}
          <Card>
            <h3 className="text-xs font-semibold mb-3" style={{ color: 'var(--text-muted)' }}>System Status</h3>
            <div className="space-y-2">
              {[
                { label: 'Data Source', value: 'Polygon.io', status: 'online' as const },
                { label: 'Ralph Engine', value: 'Streaming', status: 'online' as const },
                { label: 'Market Session', value: MOCK_MARKET.session, status: 'online' as const },
                { label: 'Forward Testing', value: '3 strategies', status: 'online' as const },
              ].map(item => (
                <div key={item.label} className="flex items-center justify-between">
                  <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>{item.label}</span>
                  <span
                    className="text-[10px] px-2 py-0.5 rounded-full flex items-center gap-1"
                    style={{
                      background: 'var(--green-muted)',
                      color: 'var(--green)',
                    }}
                  >
                    <span className="w-1 h-1 rounded-full" style={{ background: 'var(--green)' }} />
                    {item.value}
                  </span>
                </div>
              ))}
              <div className="flex items-center justify-between pt-1 mt-1 border-t" style={{ borderColor: 'var(--border)' }}>
                <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>Last Update</span>
                <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>10:32:15 AM</span>
              </div>
            </div>
          </Card>

          {/* Quick Actions */}
          <div className="grid grid-cols-2 gap-2">
            {[
              { label: 'Build', href: '/strategy-builder', icon: '+' },
              { label: 'Portfolios', href: '/portfolios', icon: 'P' },
              { label: 'Alerts', href: '/alerts', icon: 'A' },
              { label: 'Packs', href: '/confluence-packs', icon: 'C' },
            ].map(action => (
              <Link key={action.label} href={action.href}>
                <div
                  className="rounded-lg p-2.5 text-center transition-colors cursor-pointer"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
                >
                  <span
                    className="text-sm font-bold block mb-0.5"
                    style={{ color: 'var(--accent)' }}
                  >
                    {action.icon}
                  </span>
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{action.label}</span>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </div>

      {/* ============ NOTIFICATION CENTER MODAL ============ */}
      <Modal
        title="Notifications"
        isOpen={showNotifications}
        onClose={() => setShowNotifications(false)}
        width="420px"
      >
        <div className="space-y-0">
          {MOCK_ALERTS.map(alert => {
            const typeStyle = alertTypeColors[alert.type] || alertTypeColors.system;
            return (
              <div
                key={alert.id}
                className="flex items-start gap-3 py-3 border-b last:border-0"
                style={{
                  borderColor: 'var(--border)',
                  background: !alert.read ? 'color-mix(in srgb, var(--accent) 5%, transparent)' : 'transparent',
                }}
              >
                <div
                  className="w-8 h-8 rounded-lg flex items-center justify-center text-xs font-bold flex-shrink-0 mt-0.5"
                  style={{ background: typeStyle.bg, color: typeStyle.color }}
                >
                  {typeStyle.icon}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>{alert.message}</p>
                  <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>{alert.time}</p>
                </div>
                {!alert.read && (
                  <span className="w-2 h-2 rounded-full flex-shrink-0 mt-2" style={{ background: 'var(--accent)' }} />
                )}
              </div>
            );
          })}
        </div>
        <button
          className="w-full mt-3 py-2 rounded-lg text-xs font-medium"
          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-muted)' }}
        >
          Mark All as Read
        </button>
      </Modal>
    </div>
  );
}
