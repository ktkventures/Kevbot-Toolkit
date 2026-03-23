'use client';

import { useState, useMemo } from 'react';
import Link from 'next/link';
import Card from '@/components/Card';
import Modal from '@/components/Modal';

// =============================================================================
// V5 Meta: Hybrid Cockpit — V1 Chart Focus + V4 Creative Widgets
// =============================================================================
// Combines V1's clean chart-centric layout (equity curve + daily P&L as hero
// content) with V4's best widgets: active positions with match-status and
// close-early, market regime with VIX, monthly goal tracker, P&L calendar
// heatmap. New additions: issues/warnings panel for anomaly detection,
// position match status (backtest vs live), and customizable widget toggles.
// =============================================================================

// ---------------------------------------------------------------------------
// CSS Animations
// ---------------------------------------------------------------------------

const ANIMATION_STYLES = `
@keyframes v5-fade-in {
  0% { opacity: 0; transform: translateY(6px); }
  100% { opacity: 1; transform: translateY(0); }
}
@keyframes v5-glow-border {
  0%, 100% { box-shadow: 0 0 4px rgba(0,255,136,0.08); }
  50% { box-shadow: 0 0 14px rgba(0,255,136,0.22); }
}
@keyframes v5-pulse-dot {
  0%, 100% { opacity: 0.4; transform: scale(1); }
  50% { opacity: 1; transform: scale(1.4); }
}
@keyframes v5-progress-fill {
  0% { width: 0%; }
}
@keyframes v5-regime-pulse {
  0%, 100% { opacity: 0.6; }
  50% { opacity: 1; }
}
@keyframes v5-slide-in {
  0% { transform: translateX(12px); opacity: 0; }
  100% { transform: translateX(0); opacity: 1; }
}
`;

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
// Default widget config
// ---------------------------------------------------------------------------

interface WidgetConfig {
  id: string;
  label: string;
  enabled: boolean;
}

const DEFAULT_WIDGETS: WidgetConfig[] = [
  { id: 'equity-curve', label: 'Portfolio Equity Curve', enabled: true },
  { id: 'daily-pnl', label: 'Daily P&L Bar Chart', enabled: true },
  { id: 'positions', label: 'Active Positions', enabled: true },
  { id: 'monthly-goal', label: 'Monthly Goal Tracker', enabled: true },
  { id: 'market-regime', label: 'Market Regime', enabled: true },
  { id: 'calendar', label: 'P&L Calendar Heatmap', enabled: true },
  { id: 'activity', label: 'Recent Activity Feed', enabled: true },
  { id: 'issues', label: 'Issues & Warnings', enabled: true },
  { id: 'quick-actions', label: 'Quick Actions', enabled: true },
  { id: 'system-status', label: 'System Status', enabled: true },
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
        <linearGradient id="v5-equityGrad" x1="0" y1="0" x2="0" y2="1">
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
      <path d={areaD} fill="url(#v5-equityGrad)" />

      {/* Line */}
      <path d={lineD} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />

      {/* End dot */}
      <circle cx={lastPt.x} cy={lastPt.y} r="4" fill={color} style={{ animation: 'v5-pulse-dot 2s ease-in-out infinite' }} />
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
          style={{ animation: 'v5-regime-pulse 3s ease-in-out infinite' }}
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
            animation: 'v5-progress-fill 1s ease-out',
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

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function DashboardV5() {
  const [showCustomize, setShowCustomize] = useState(false);
  const [widgets, setWidgets] = useState<WidgetConfig[]>(DEFAULT_WIDGETS);

  const isEnabled = (id: string) => widgets.find(w => w.id === id)?.enabled ?? true;

  const toggleWidget = (id: string) => {
    setWidgets(prev => prev.map(w => w.id === id ? { ...w, enabled: !w.enabled } : w));
  };

  const alertTypeStyles: Record<string, { bg: string; color: string; icon: string }> = {
    entry: { bg: 'var(--green-muted)', color: 'var(--green)', icon: '\u2191' },
    exit: { bg: 'var(--red-muted)', color: 'var(--red)', icon: '\u2193' },
    system: { bg: 'rgba(156,163,175,0.15)', color: 'var(--text-muted)', icon: 'S' },
  };

  const severityStyles: Record<string, { bg: string; color: string; icon: string }> = {
    warning: { bg: 'var(--orange-muted)', color: 'var(--orange)', icon: '!' },
    error: { bg: 'var(--red-muted)', color: 'var(--red)', icon: '\u00D7' },
    info: { bg: 'var(--blue-muted)', color: 'var(--blue)', icon: 'i' },
  };

  const systemHealthy = MOCK_SYSTEM.polygon && MOCK_SYSTEM.ralph;

  return (
    <div style={{ animation: 'v5-fade-in 0.3s ease-out' }}>
      <style>{ANIMATION_STYLES}</style>

      {/* ============ HEADER: Title + Customize ============ */}
      <div className="flex items-start justify-between mb-5">
        <div>
          <h1 className="text-2xl font-bold mb-1">Dashboard</h1>
          <span className="text-sm" style={{ color: 'var(--text-muted)' }}>
            Trading Cockpit
          </span>
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

      {/* ============ TOP METRICS STRIP ============ */}
      <div className="grid grid-cols-5 gap-3 mb-5">
        {/* Today P&L */}
        <div
          className="rounded-lg border p-3"
          style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}
        >
          <p className="text-[10px] mb-0.5" style={{ color: 'var(--text-muted)' }}>Today P&L</p>
          <div className="flex items-baseline gap-1.5">
            <span className="text-lg font-semibold" style={{ color: 'var(--green)' }}>
              ${MOCK_PORTFOLIO.todayPnL}
            </span>
            <span className="text-xs font-mono" style={{ color: 'var(--green)' }}>
              +{MOCK_PORTFOLIO.todayPnLPct.toFixed(2)}%
            </span>
          </div>
        </div>

        {/* Profit Factor */}
        <div
          className="rounded-lg border p-3"
          style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}
        >
          <p className="text-[10px] mb-0.5" style={{ color: 'var(--text-muted)' }}>Profit Factor</p>
          <span className="text-lg font-semibold">{MOCK_PORTFOLIO.profitFactor.toFixed(2)}</span>
        </div>

        {/* Open Positions */}
        <div
          className="rounded-lg border p-3"
          style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}
        >
          <p className="text-[10px] mb-0.5" style={{ color: 'var(--text-muted)' }}>Open Positions</p>
          <div className="flex items-center gap-2">
            <span className="text-lg font-semibold">{MOCK_PORTFOLIO.openPositions}</span>
            <span
              className="text-[9px] px-1.5 py-0.5 rounded-full font-medium"
              style={{ background: 'var(--green-muted)', color: 'var(--green)' }}
            >
              Live
            </span>
          </div>
        </div>

        {/* Market Regime */}
        <div
          className="rounded-lg border p-3"
          style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}
        >
          <p className="text-[10px] mb-0.5" style={{ color: 'var(--text-muted)' }}>Market Regime</p>
          <div className="flex items-center gap-2">
            <span className="text-lg font-semibold" style={{ color: 'var(--green)' }}>
              {MOCK_MARKET.regime}
            </span>
            <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
              VIX {MOCK_MARKET.vix}
            </span>
          </div>
        </div>

        {/* System Status */}
        <div
          className="rounded-lg border p-3"
          style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}
        >
          <p className="text-[10px] mb-0.5" style={{ color: 'var(--text-muted)' }}>System Status</p>
          <div className="flex items-center gap-2">
            <span
              className="w-2.5 h-2.5 rounded-full"
              style={{
                background: systemHealthy ? 'var(--green)' : 'var(--red)',
                boxShadow: systemHealthy ? '0 0 6px rgba(76,175,80,0.5)' : '0 0 6px rgba(244,67,54,0.5)',
              }}
            />
            <span className="text-lg font-semibold" style={{ color: systemHealthy ? 'var(--green)' : 'var(--red)' }}>
              {systemHealthy ? 'Healthy' : 'Issues'}
            </span>
          </div>
        </div>
      </div>

      {/* ============ MAIN CONTENT: Two-column layout ============ */}
      <div className="grid grid-cols-12 gap-4 mb-4">
        {/* -- LEFT COLUMN: Charts (7 cols ~ 60%) -- */}
        <div className="col-span-7 space-y-4">
          {/* Portfolio Equity Curve */}
          {isEnabled('equity-curve') && (
            <Card>
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>
                  Portfolio Equity Curve
                </h3>
                <span className="text-[10px] font-mono" style={{ color: 'var(--green)' }}>
                  +${EQUITY_DATA[EQUITY_DATA.length - 1].toLocaleString()} cumulative
                </span>
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
              </div>
              <DailyPnLChart data={DAILY_PNL} width={700} height={220} />
            </Card>
          )}
        </div>

        {/* -- RIGHT COLUMN: Widgets (5 cols ~ 40%) -- */}
        <div className="col-span-5 space-y-4">
          {/* Active Positions */}
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
                    animation: `v5-glow-border 3s ease-in-out infinite, v5-slide-in 0.3s ease-out ${idx * 0.1}s both`,
                  }}
                >
                  {/* Top row: symbol, direction, match status, close button */}
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

          {/* Monthly Goal Tracker */}
          {isEnabled('monthly-goal') && (
            <Card>
              <h3 className="text-xs font-semibold mb-3" style={{ color: 'var(--text-muted)' }}>Monthly Goal</h3>
              <GoalProgress
                current={MOCK_PORTFOLIO.monthlyProgress}
                target={MOCK_PORTFOLIO.monthlyTarget}
              />
            </Card>
          )}

          {/* Market Regime */}
          {isEnabled('market-regime') && (
            <Card>
              <h3 className="text-xs font-semibold mb-2" style={{ color: 'var(--text-muted)' }}>Market Regime</h3>
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

          {/* P&L Calendar Heatmap */}
          {isEnabled('calendar') && (
            <Card>
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>March 2026 P&L</h3>
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
              <PnLCalendar data={CALENDAR_DATA} currentDay={22} />
            </Card>
          )}
        </div>
      </div>

      {/* ============ BOTTOM ROW: Full-width widgets ============ */}
      <div className="grid grid-cols-12 gap-4 mb-4">
        {/* Recent Activity Feed */}
        {isEnabled('activity') && (
          <div className="col-span-6">
            <Card>
              <h3 className="text-xs font-semibold mb-3" style={{ color: 'var(--text-muted)' }}>Recent Activity</h3>
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
          </div>
        )}

        {/* Issues & Warnings */}
        {isEnabled('issues') && (
          <div className="col-span-6">
            <Card>
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>
                  Issues & Warnings
                </h3>
                <span
                  className="text-[9px] px-1.5 py-0.5 rounded-full font-medium"
                  style={{
                    background: MOCK_ISSUES.length > 0 ? 'var(--orange-muted)' : 'var(--green-muted)',
                    color: MOCK_ISSUES.length > 0 ? 'var(--orange)' : 'var(--green)',
                  }}
                >
                  {MOCK_ISSUES.length} active
                </span>
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
          </div>
        )}
      </div>

      {/* ============ BOTTOM UTILITY ROW ============ */}
      <div className="grid grid-cols-12 gap-4">
        {/* Quick Actions */}
        {isEnabled('quick-actions') && (
          <div className="col-span-8">
            <div className="flex gap-2">
              {[
                { label: 'Strategy Builder', href: '/strategy-builder', icon: '+' },
                { label: 'Portfolios', href: '/portfolios', icon: 'P' },
                { label: 'Alerts', href: '/alerts', icon: 'A' },
                { label: 'Settings', href: '/settings', icon: 'S' },
              ].map(action => (
                <Link key={action.label} href={action.href} className="flex-1">
                  <div
                    className="rounded-lg p-3 text-center transition-colors cursor-pointer"
                    style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}
                  >
                    <span className="text-sm font-bold block mb-0.5" style={{ color: 'var(--accent)' }}>
                      {action.icon}
                    </span>
                    <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{action.label}</span>
                  </div>
                </Link>
              ))}
            </div>
          </div>
        )}

        {/* System Status (compact) */}
        {isEnabled('system-status') && (
          <div className="col-span-4">
            <Card>
              <h3 className="text-xs font-semibold mb-2" style={{ color: 'var(--text-muted)' }}>System</h3>
              <div className="space-y-1.5">
                {[
                  { label: 'Polygon.io', ok: MOCK_SYSTEM.polygon },
                  { label: 'Ralph Engine', ok: MOCK_SYSTEM.ralph },
                ].map(item => (
                  <div key={item.label} className="flex items-center justify-between">
                    <span className="text-[11px]" style={{ color: 'var(--text-secondary)' }}>{item.label}</span>
                    <span
                      className="text-[10px] px-1.5 py-0.5 rounded-full flex items-center gap-1"
                      style={{
                        background: item.ok ? 'var(--green-muted)' : 'var(--red-muted)',
                        color: item.ok ? 'var(--green)' : 'var(--red)',
                      }}
                    >
                      <span
                        className="w-1 h-1 rounded-full"
                        style={{ background: item.ok ? 'var(--green)' : 'var(--red)' }}
                      />
                      {item.ok ? 'Connected' : 'Down'}
                    </span>
                  </div>
                ))}
                <div className="flex items-center justify-between pt-1 border-t" style={{ borderColor: 'var(--border)' }}>
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Uptime</span>
                  <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>
                    {MOCK_SYSTEM.workerUptime}
                  </span>
                </div>
              </div>
            </Card>
          </div>
        )}
      </div>

      {/* ============ CUSTOMIZE MODAL ============ */}
      <Modal
        title="Customize Dashboard"
        isOpen={showCustomize}
        onClose={() => setShowCustomize(false)}
        width="480px"
      >
        <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
          Toggle widgets on or off. Widget sizing and positioning coming soon.
        </p>
        <div className="space-y-0">
          {widgets.map(widget => (
            <div
              key={widget.id}
              className="flex items-center justify-between py-3 border-b last:border-0"
              style={{ borderColor: 'var(--border)' }}
            >
              <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                {widget.label}
              </span>
              <button
                className="relative w-10 h-5 rounded-full transition-colors"
                style={{
                  background: widget.enabled ? 'var(--accent)' : 'var(--bg-input)',
                  border: '1px solid',
                  borderColor: widget.enabled ? 'var(--accent)' : 'var(--border)',
                }}
                onClick={() => toggleWidget(widget.id)}
              >
                <span
                  className="absolute top-0.5 w-3.5 h-3.5 rounded-full transition-all"
                  style={{
                    background: widget.enabled ? '#000' : 'var(--text-muted)',
                    left: widget.enabled ? '22px' : '3px',
                  }}
                />
              </button>
            </div>
          ))}
        </div>
      </Modal>
    </div>
  );
}
