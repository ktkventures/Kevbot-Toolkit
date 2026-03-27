'use client';

/**
 * Dashboard — Faithful copy of V7 design with mock data replaced by API hooks.
 * Source: V7 cockpit layout (2,492 lines)
 *
 * Data convention:
 *   Real value = live from API
 *   -- = wired but no data yet
 *   {{field}} = not wired, needs backend work
 */

import { useState, useRef, useEffect, useMemo } from 'react';
import Link from 'next/link';
import Card from '@/components/Card';
import Modal from '@/components/Modal';
import { useDashboardSummary, useDashboardEquityCurve, useDashboardDailyPnl, useDashboardMarketRegime } from '@/hooks/queries/useDashboard';
import { useStrategies } from '@/hooks/queries/useStrategies';
import { useMonitorStatus, useEngineState } from '@/hooks/queries/useAlerts';
import { usePortfolios } from '@/hooks/queries/usePortfolios';

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
// Types for derived dashboard data
// ---------------------------------------------------------------------------

interface DashboardPosition {
  id: string;
  strategy: string;
  symbol: string;
  direction: 'LONG' | 'SHORT';
  entryPrice: number;
  currentPrice: number;
  unrealizedR: number;
  entryTime: string;
  matched: boolean;
}

interface HealthItem {
  id: string;
  name: string;
  deviationSd: number;
  status: 'on_track' | 'outperforming' | 'underperforming';
  alertTrades: number;
  expectedR?: number;
  actualR?: number;
  expectedPnL?: number;
  actualPnL?: number;
}

interface DashboardIssue {
  id: string;
  severity: 'warning' | 'error' | 'info';
  message: string;
  link: string;
}

interface ActivityItem {
  time: string;
  type: 'entry' | 'exit' | 'system' | 'summary';
  message: string;
}

// ---------------------------------------------------------------------------
// Empty defaults for Phase C features (equity, P&L, calendar, market, goals)
// These will be populated when dedicated API endpoints are built.
// ---------------------------------------------------------------------------

const EMPTY_EQUITY_SEGMENTS = { btTotalR: 0, fwTotalR: 0, liveTotalR: 0 };
const EMPTY_EQUITY_INFLECTIONS: { label: string; value: string; date: string }[] = [];
const EMPTY_PNL_STATS = {
  totalDays: 0, winDays: 0, lossDays: 0, avgWinDay: 0, avgLossDay: 0,
  bestDay: { value: 0, date: '--' }, worstDay: { value: 0, date: '--' },
  currentStreak: { type: 'win' as const, days: 0 },
};
const EMPTY_PNL_WEEKLY: { week: string; pnl: number; trades: number; winRate: number }[] = [];
const EMPTY_MONTHLY_SUMMARY = {
  prevMonthPnL: 0, prevMonthWinRate: 0, prevMonthTradingDays: 0,
  consistencyScore: 0,
  consistencyExplanation: 'Consistency score measures the % of trading days that ended green, weighted by how close each day\'s P&L was to your average. A score above 60 indicates reliable execution.',
  goalProgress: { target: 0, current: 0, percentComplete: 0 },
};
const EMPTY_VIX_HISTORY: { day: string; value: number }[] = [];
const EMPTY_REGIME_HISTORY: { regime: string; startDate: string; endDate: string; days: number }[] = [];
const EMPTY_GOAL_WEEKLY_PROGRESS: { week: string; earned: number; target: number; pace: 'ahead' | 'behind' }[] = [];
const EMPTY_GOAL_HISTORY: { month: string; target: number; achieved: number; hit: boolean }[] = [];
const EMPTY_RESOLVED_ISSUES: { id: string; severity: 'warning' | 'error' | 'info'; message: string; resolvedAt: string }[] = [];
const EMPTY_POSITION_DETAILS: Record<string, {
  strategyId: string; entryTrigger: string; confluenceConditions: string[];
  stopLevel: number; targetLevel: number; avgDuration: string; timeInTrade: string;
  btWinRate: number; btAvgR: number; recentTrades: { date: string; result: string; r: number }[];
}> = {};
const EMPTY_HEALTH_DETAILS: Record<string, {
  explanation: string; recommendation: string; healthyLastWeek: boolean;
  correlations: string[]; expectedRHistory: number[]; actualRHistory: number[];
}> = {};

// ---------------------------------------------------------------------------
// KPI config builder — values derived from API data at render time
// ---------------------------------------------------------------------------

function buildKpis(summary: {
  strategy_count: number;
  portfolio_count: number;
  monitored_count: number;
  total_trades: number;
  total_r: number;
  avg_win_rate: number;
} | null, positionCount: number, monitorRunning: boolean): KpiConfig[] {
  const s = summary;
  const totalR = s?.total_r ?? 0;
  const totalTrades = s?.total_trades ?? 0;
  return [
    { id: 'active-strategies', label: 'Active Strategies', value: s ? String(s.monitored_count || s.strategy_count) : '--', enabled: true },
    { id: 'total-strategies', label: 'Total Strategies', value: s ? String(s.strategy_count) : '--', enabled: true },
    { id: 'open-positions', label: 'Open Positions', value: String(positionCount), enabled: true },
    { id: 'win-rate', label: 'Win Rate', value: s ? `${s.avg_win_rate.toFixed(1)}%` : '--', enabled: true },
    { id: 'pf', label: 'Profit Factor', value: s && totalTrades > 0 ? `${((s.avg_win_rate / 100 * 2) / Math.max(1 - s.avg_win_rate / 100, 0.01)).toFixed(2)}` : '--', enabled: true },
    { id: 'total-r', label: 'Total R', value: s ? `${totalR >= 0 ? '+' : ''}${totalR.toFixed(1)}R` : '--', enabled: true },
    { id: 'daily-r', label: 'Avg Daily R', value: s && totalTrades > 0 ? `${(totalR / Math.max(totalTrades, 1)).toFixed(2)}R` : '--', enabled: false },
    { id: 'total-trades', label: 'Total Trades', value: s ? String(totalTrades) : '--', enabled: false },
    { id: 'market-regime', label: 'Market Regime', value: monitorRunning ? 'Live' : '--', enabled: false },
    { id: 'alerts-today', label: 'Alerts Today', value: '--', enabled: false },
    { id: 'balance', label: 'Balance', value: '--', enabled: false },
  ];
}

// ---------------------------------------------------------------------------
// Portfolio options — built from API data at render time
// ---------------------------------------------------------------------------

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
  if (data.length < 2) {
    return (
      <div className="flex items-center justify-center" style={{ height, color: 'var(--text-muted)' }}>
        <span className="text-xs">No equity data available yet</span>
      </div>
    );
  }

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
  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center" style={{ height, color: 'var(--text-muted)' }}>
        <span className="text-xs">No P&L data available yet</span>
      </div>
    );
  }

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
  if (target <= 0) {
    return (
      <div className="text-xs text-center py-3" style={{ color: 'var(--text-muted)' }}>Set monthly R target in Settings</div>
    );
  }
  const pct = target > 0 ? Math.max(0, Math.min((current / target) * 100, 150)) : 0;
  const now = new Date();
  const dayOfMonth = now.getDate();
  const daysInMonth = new Date(now.getFullYear(), now.getMonth() + 1, 0).getDate();
  const daysRemaining = daysInMonth - dayOfMonth;
  const onTrack = pct >= (dayOfMonth / daysInMonth) * 100;
  const dailyNeeded = daysRemaining > 0 ? ((target - current) / daysRemaining) : 0;

  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>Monthly R Target</span>
        <span className="text-xs font-mono" style={{ color: onTrack ? 'var(--green)' : 'var(--orange)' }}>
          {Math.round(pct)}%
        </span>
      </div>
      <div className="relative h-2.5 rounded-full overflow-hidden" style={{ background: 'var(--bg-input)' }}>
        <div
          className="h-full rounded-full"
          style={{
            width: `${Math.min(pct, 100)}%`,
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
          {current >= 0 ? '+' : ''}{current.toFixed(1)}R / {target.toFixed(0)}R
        </span>
        <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
          {onTrack ? 'On track' : `Need ${dailyNeeded.toFixed(1)}R/day`}
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
  if (data.length < 2) {
    return <svg width={width} height={height} />;
  }
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

export default function DashboardPage() {
  // =========================================================================
  // API hooks — ALL hooks must come before any early returns
  // =========================================================================
  const { data: dashboardSummary, isLoading: summaryLoading, error: summaryError } = useDashboardSummary();
  const { data: apiStrategies, isLoading: strategiesLoading } = useStrategies();
  const { data: monitorStatus, isLoading: monitorLoading } = useMonitorStatus();
  const { data: engineState, isLoading: engineLoading } = useEngineState();
  const { data: apiPortfolios, isLoading: portfoliosLoading } = usePortfolios();
  const { data: equityCurveData } = useDashboardEquityCurve();
  const { data: dailyPnlData } = useDashboardDailyPnl();
  const { data: marketRegimeData } = useDashboardMarketRegime();

  // Settings loaded inline to avoid adding another hook import
  // (useSettings from queries/useSettings caused circular dep in production bundle)
  const [userSettings, setUserSettings] = useState<Record<string, any> | null>(null);
  useEffect(() => {
    const token = typeof window !== 'undefined' ? localStorage.getItem('ror_access_token') : null;
    if (!token) return;
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    fetch(`${apiUrl}/api/settings`, { headers: { Authorization: `Bearer ${token}` } })
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (d) setUserSettings(d); })
      .catch(() => {});
  }, []);

  // =========================================================================
  // UI state
  // =========================================================================
  const [showCustomize, setShowCustomize] = useState(false);
  const [customizeTab, setCustomizeTab] = useState<'KPIs' | 'Widgets' | 'Quick Actions'>('Widgets');
  const [healthViewTab, setHealthViewTab] = useState<'portfolios' | 'strategies'>('portfolios');
  const [widgets, setWidgets] = useState<WidgetConfig[]>(DEFAULT_WIDGETS);
  const [portfolioSelections, setPortfolioSelections] = useState<Record<string, boolean>>({});

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

  // =========================================================================
  // Derive data from API responses
  // =========================================================================

  // Positions — from engine state, or empty if engine not running
  const positions: DashboardPosition[] = useMemo(() => {
    if (!engineState || !engineState.positions) return [];
    try {
      const posMap = engineState.positions as Record<string, any>;
      return Object.entries(posMap)
        .filter(([, p]) => p.status === 'IN_POSITION')
        .map(([key, p]) => ({
          id: key,
          strategy: p.strategy_name || key,
          symbol: p.symbol || '--',
          direction: (p.direction || 'LONG') as 'LONG' | 'SHORT',
          entryPrice: p.entry_price ?? 0,
          currentPrice: p.current_price ?? p.entry_price ?? 0,
          unrealizedR: p.unrealized_r ?? 0,
          entryTime: p.entry_time || '--',
          matched: p.matched !== false,
        }));
    } catch {
      return [];
    }
  }, [engineState]);

  // System status — derived from monitor status
  const systemStatus = useMemo(() => {
    const ralph = monitorStatus?.running === true;
    const polygon = monitorStatus?.polygon_connected !== false; // {{polygon_status}} — assume ok unless explicitly false
    return { polygon, ralph, workerUptime: monitorStatus?.uptime || '--', lastUpdate: monitorStatus?.last_update || '--' };
  }, [monitorStatus]);

  // Health strategies — derived from API strategies with basic KPI health
  const healthStrategies: HealthItem[] = useMemo(() => {
    if (!apiStrategies) return [];
    return apiStrategies
      .filter(s => s.kpis && Object.keys(s.kpis).length > 0)
      .slice(0, 10)
      .map(s => {
        const k = s.kpis || {};
        const sigma = (s as any).sigma_fwd ?? (s as any).sigma_alert ?? 0;
        const rawStatus = (s as any).status || '';
        const status: 'on_track' | 'outperforming' | 'underperforming' | 'insufficient_data' =
          rawStatus.includes('Outperform') ? 'outperforming' :
          rawStatus.includes('Underperform') ? 'underperforming' :
          rawStatus.includes('Insufficient') ? 'insufficient_data' : 'on_track';
        return {
          id: String(s.id),
          name: s.name || '--',
          deviationSd: typeof sigma === 'number' ? sigma : 0,
          status,
          alertTrades: (s as any).forward_kpis?.trades ?? k.total_trades ?? 0,
          expectedR: k.avg_r ?? k.daily_r ?? 0,
          actualR: (s as any).forward_kpis?.total_r ?? k.total_r ?? 0,
        };
      });
  }, [apiStrategies]);

  // Health portfolios — {{portfolio_health}} — needs backend compute
  const healthPortfolios: HealthItem[] = useMemo(() => {
    if (!apiPortfolios) return [];
    return apiPortfolios.map(p => ({
      id: String(p.id),
      name: p.name || '--',
      deviationSd: 0,
      status: 'on_track' as const,
      alertTrades: 0,
      expectedPnL: 0,
      actualPnL: 0,
    }));
  }, [apiPortfolios]);

  // Portfolio filter options — from API
  const portfolios: PortfolioOption[] = useMemo(() => {
    if (!apiPortfolios) return [];
    return apiPortfolios.map(p => ({
      id: String(p.id),
      name: p.name || 'Unnamed',
      selected: portfolioSelections[String(p.id)] !== false, // default selected
    }));
  }, [apiPortfolios, portfolioSelections]);

  // Issues — empty array, needs dedicated endpoint {{issues}}
  const issues: DashboardIssue[] = [];

  // Activity feed — empty array, needs activity/alert log endpoint {{activity}}
  const activityFeed: ActivityItem[] = [];

  // Market data — wired to /api/dashboard/market-regime
  const marketData = useMemo(() => ({
    regime: marketRegimeData?.regime ?? '--',
    vix: marketRegimeData?.vix ?? 0,
    spy: {
      price: marketRegimeData?.spy?.price ?? 0,
      change: marketRegimeData?.spy?.change ?? 0,
      changePct: marketRegimeData?.spy?.change_pct ?? 0,
    },
    qqq: {
      price: (marketRegimeData as any)?.qqq?.price ?? 0,
      change: (marketRegimeData as any)?.qqq?.change ?? 0,
      changePct: (marketRegimeData as any)?.qqq?.change_pct ?? 0,
    },
    breadth: null, // No data source available — hidden in UI
    vixLabel: (marketRegimeData as any)?.vix_label ?? 'VIX',
  }), [marketRegimeData]);

  // Equity curve — wired to /api/dashboard/equity-curve
  const equityData: number[] = useMemo(() => {
    if (!equityCurveData?.points?.length) return [];
    return equityCurveData.points.map(p => p.value);
  }, [equityCurveData]);
  // Daily P&L — wired to /api/dashboard/daily-pnl
  const dailyPnl: { day: string; value: number }[] = useMemo(() => {
    if (!dailyPnlData?.days?.length) return [];
    return dailyPnlData.days;
  }, [dailyPnlData]);
  // Calendar — derived from daily P&L for current month
  const calendarData: Record<number, number> = useMemo(() => {
    if (!dailyPnlData?.days?.length) return {};
    const now = new Date();
    const currentMonth = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`;
    const result: Record<number, number> = {};
    for (const d of dailyPnlData.days) {
      if (d.day.startsWith(currentMonth)) {
        const dayNum = parseInt(d.day.split('-')[2], 10);
        result[dayNum] = Math.round(d.value * 100) / 100;
      }
    }
    return result;
  }, [dailyPnlData]);
  // Monthly goal — from settings + current month P&L from dailyPnl
  const monthlyTarget = useMemo(() => {
    return (userSettings as any)?.monthly_r_target ?? (userSettings as any)?.monthlyRTarget ?? 10;
  }, [userSettings]);
  const monthlyProgress = useMemo(() => {
    if (!dailyPnlData?.days?.length) return 0;
    const now = new Date();
    const currentMonth = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`;
    return dailyPnlData.days
      .filter(d => d.day.startsWith(currentMonth))
      .reduce((sum, d) => sum + d.value, 0);
  }, [dailyPnlData]);

  // KPIs — built from summary + derived data
  const [kpis, setKpis] = useState<KpiConfig[]>([]);
  useEffect(() => {
    setKpis(buildKpis(dashboardSummary || null, positions.length, systemStatus.ralph));
  }, [dashboardSummary, positions.length, systemStatus.ralph]);

  const isEnabled = (id: string) => widgets.find(w => w.id === id)?.enabled ?? true;

  const toggleWidget = (id: string) => {
    setWidgets(prev => prev.map(w => w.id === id ? { ...w, enabled: !w.enabled } : w));
  };

  const toggleKpi = (id: string) => {
    setKpis(prev => prev.map(k => k.id === id ? { ...k, enabled: !k.enabled } : k));
  };

  const togglePortfolio = (id: string) => {
    setPortfolioSelections(prev => {
      const current = prev[id] !== false; // default to true (selected)
      return { ...prev, [id]: !current };
    });
  };

  const enabledKpis = kpis.filter(k => k.enabled);
  const selectedPortfolioCount = portfolios.filter(p => p.selected).length;

  const systemHealthy = systemStatus.polygon && systemStatus.ralph;
  const issueCount = issues.length;

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
  const filteredActivity = activityFeed
    .filter(a => activityFilter === 'all' || a.type === activityFilter)
    .filter(a => !activitySearch || a.message.toLowerCase().includes(activitySearch.toLowerCase()));

  // Inject animation CSS (must be before early returns — React hooks rule)
  useEffect(() => {
    const id = 'v7-animation-styles';
    if (!document.getElementById(id)) {
      const style = document.createElement('style');
      style.id = id;
      style.textContent = ANIMATION_STYLES;
      document.head.appendChild(style);
    }
  }, []);

  // =========================================================================
  // Loading / Error states
  // =========================================================================
  const isLoading = summaryLoading && strategiesLoading && portfoliosLoading;

  if (isLoading) {
    return (
      <div style={{ animation: 'v7-fade-in 0.3s ease-out' }}>
        <h1 className="text-2xl font-bold mb-1">Dashboard</h1>
        <span className="text-sm" style={{ color: 'var(--text-muted)' }}>Trading Cockpit</span>
        <div className="mt-8 space-y-4">
          {[1, 2, 3].map(i => (
            <div key={i} className="h-24 rounded-lg animate-pulse" style={{ background: 'var(--bg-input)' }} />
          ))}
        </div>
      </div>
    );
  }

  if (summaryError) {
    return (
      <div style={{ animation: 'v7-fade-in 0.3s ease-out' }}>
        <h1 className="text-2xl font-bold mb-1">Dashboard</h1>
        <span className="text-sm" style={{ color: 'var(--text-muted)' }}>Trading Cockpit</span>
        <Card>
          <p className="text-sm py-8 text-center" style={{ color: 'var(--red)' }}>
            Failed to load dashboard data. Please try again.
          </p>
        </Card>
      </div>
    );
  }

  return (
    <div style={{ animation: 'v7-fade-in 0.3s ease-out' }}>

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
            title={`Polygon: ${systemStatus.polygon ? 'Connected' : 'Down'} | Ralph: ${systemStatus.ralph ? 'Running' : 'Stopped'} | Uptime: ${systemStatus.workerUptime}`}
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
                    {equityData.length > 0 ? `+$${equityData[equityData.length - 1].toLocaleString()} cumulative` : '--'}
                  </span>
                  <DetailButton onClick={() => setShowEquityDetail(true)} />
                </div>
              </div>
              <EquityCurve data={equityData} width={700} height={260} />
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
              <DailyPnLChart data={dailyPnl} width={700} height={220} />
            </Card>
          )}

          {/* P&L Calendar Heatmap — under Daily P&L */}
          {isEnabled('calendar') && (
            <Card>
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>P&L Calendar</h3>
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
              <PnLCalendar data={calendarData} currentDay={new Date().getDate()} />
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
                  Active Positions ({positions.length})
                </h3>
                <span
                  className="flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full"
                  style={{ background: 'var(--green-muted)', color: 'var(--green)' }}
                >
                  <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--green)' }} />
                  Live
                </span>
              </div>
              {positions.map((pos, idx) => (
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
              {positions.length === 0 && (
                <p className="text-xs text-center py-4" style={{ color: 'var(--text-muted)' }}>
                  {systemStatus.ralph ? 'No open positions' : 'No open positions \u2014 engine not running'}
                </p>
              )}
            </Card>
          )}

          {/* Portfolio Health — SD deviation widget with tabs */}
          {isEnabled('health') && (() => {
            const [healthTab, setHealthTab] = [healthViewTab, setHealthViewTab];
            const items = healthTab === 'portfolios' ? healthPortfolios : healthStrategies;

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
                <RegimeGauge regime={(marketData.regime === 'Bull' || marketData.regime === 'Bear') ? marketData.regime : 'Neutral'} size={90} />
                <div className="flex-1 space-y-1.5">
                  <div className="flex items-center justify-between text-xs">
                    <span style={{ color: 'var(--text-muted)' }}>VIX</span>
                    <span className="font-mono" style={{ color: marketData.vix < 20 ? 'var(--green)' : 'var(--red)' }}>
                      {marketData.vix.toFixed(1)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span style={{ color: 'var(--text-muted)' }}>{(marketData as any).vixLabel || 'VIX'}</span>
                    <span className="font-mono" style={{ color: 'var(--text-secondary)' }}>
                      {marketData.vix > 0 ? `$${marketData.vix.toFixed(2)}` : '--'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span style={{ color: 'var(--text-muted)' }}>SPY</span>
                    <span className="font-mono">
                      {marketData.spy.price}{' '}
                      <span style={{ color: marketData.spy.changePct >= 0 ? 'var(--green)' : 'var(--red)' }}>
                        {marketData.spy.changePct >= 0 ? '+' : ''}{marketData.spy.changePct.toFixed(2)}%
                      </span>
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span style={{ color: 'var(--text-muted)' }}>QQQ</span>
                    <span className="font-mono">
                      {marketData.qqq.price}{' '}
                      <span style={{ color: marketData.qqq.changePct >= 0 ? 'var(--green)' : 'var(--red)' }}>
                        {marketData.qqq.changePct >= 0 ? '+' : ''}{marketData.qqq.changePct.toFixed(2)}%
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
                current={monthlyProgress}
                target={monthlyTarget}
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
                      background: issues.length > 0 ? 'var(--orange-muted)' : 'var(--green-muted)',
                      color: issues.length > 0 ? 'var(--orange)' : 'var(--green)',
                    }}
                  >
                    {issues.length} active
                  </span>
                  <DetailButton onClick={() => setShowIssuesDetail(true)} />
                </div>
              </div>
              <div className="space-y-0">
                {issues.map((issue) => {
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
                {issues.length === 0 && (
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
                {activityFeed.map((alert, idx) => {
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
          <EquityCurve data={equityData} width={720} height={320} />
        </ModalSection>

        <ModalSection title="Segment Breakdown">
          <div className="grid grid-cols-3 gap-3">
            <div className="rounded-lg p-3 text-center" style={{ background: 'var(--bg-input)' }}>
              <p className="text-[10px] mb-1" style={{ color: 'var(--text-muted)' }}>Backtest Total R</p>
              <p className="text-lg font-semibold font-mono" style={{ color: 'var(--text-primary)' }}>+{EMPTY_EQUITY_SEGMENTS.btTotalR}R</p>
            </div>
            <div className="rounded-lg p-3 text-center" style={{ background: 'var(--bg-input)' }}>
              <p className="text-[10px] mb-1" style={{ color: 'var(--text-muted)' }}>Forward Test Total R</p>
              <p className="text-lg font-semibold font-mono" style={{ color: 'var(--accent)' }}>+{EMPTY_EQUITY_SEGMENTS.fwTotalR}R</p>
            </div>
            <div className="rounded-lg p-3 text-center" style={{ background: 'var(--bg-input)' }}>
              <p className="text-[10px] mb-1" style={{ color: 'var(--text-muted)' }}>Live Total R</p>
              <p className="text-lg font-semibold font-mono" style={{ color: 'var(--green)' }}>+{EMPTY_EQUITY_SEGMENTS.liveTotalR}R</p>
            </div>
          </div>
        </ModalSection>

        <ModalSection title="Key Inflection Points">
          {EMPTY_EQUITY_INFLECTIONS.map((pt, i) => (
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
          <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
            Month-over-month equity comparison will be available when P&L history endpoints are built.
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
            <StatRow label="Total Trading Days" value={String(EMPTY_PNL_STATS.totalDays)} />
            <StatRow label="Win Days" value={String(EMPTY_PNL_STATS.winDays)} color="var(--green)" />
            <StatRow label="Loss Days" value={String(EMPTY_PNL_STATS.lossDays)} color="var(--red)" />
            <StatRow label="Win Rate" value={`${Math.round((EMPTY_PNL_STATS.winDays / EMPTY_PNL_STATS.totalDays) * 100)}%`} />
            <StatRow label="Avg Win Day" value={`+$${EMPTY_PNL_STATS.avgWinDay.toFixed(0)}`} color="var(--green)" />
            <StatRow label="Avg Loss Day" value={`-$${Math.abs(EMPTY_PNL_STATS.avgLossDay).toFixed(0)}`} color="var(--red)" />
          </div>
        </ModalSection>

        <ModalSection title="Highlights">
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-lg p-3" style={{ background: 'var(--green-muted)' }}>
              <p className="text-[10px] mb-0.5" style={{ color: 'var(--text-muted)' }}>Best Day</p>
              <p className="text-sm font-semibold font-mono" style={{ color: 'var(--green)' }}>+${EMPTY_PNL_STATS.bestDay.value}</p>
              <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{EMPTY_PNL_STATS.bestDay.date}</p>
            </div>
            <div className="rounded-lg p-3" style={{ background: 'var(--red-muted)' }}>
              <p className="text-[10px] mb-0.5" style={{ color: 'var(--text-muted)' }}>Worst Day</p>
              <p className="text-sm font-semibold font-mono" style={{ color: 'var(--red)' }}>-${Math.abs(EMPTY_PNL_STATS.worstDay.value)}</p>
              <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{EMPTY_PNL_STATS.worstDay.date}</p>
            </div>
          </div>
        </ModalSection>

        <ModalSection title="Current Streak">
          <div className="flex items-center gap-2">
            <span
              className="text-xs px-2 py-1 rounded font-medium"
              style={{
                background: EMPTY_PNL_STATS.currentStreak.type === 'win' ? 'var(--green-muted)' : 'var(--red-muted)',
                color: EMPTY_PNL_STATS.currentStreak.type === 'win' ? 'var(--green)' : 'var(--red)',
              }}
            >
              {EMPTY_PNL_STATS.currentStreak.days} {EMPTY_PNL_STATS.currentStreak.type} days in a row
            </span>
          </div>
        </ModalSection>

        <ModalSection title="Weekly Breakdown">
          <div className="space-y-0">
            {EMPTY_PNL_WEEKLY.map((week, i) => (
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
        title="Monthly Summary"
        isOpen={showCalendarDetail}
        onClose={() => setShowCalendarDetail(false)}
        width="680px"
      >
        <ModalSection title="Month Stats">
          <div className="grid grid-cols-2 gap-x-6">
            <StatRow
              label="Total P&L"
              value={`+$${Object.values(calendarData).reduce((a, b) => a + b, 0).toLocaleString()}`}
              color="var(--green)"
            />
            <StatRow label="Trading Days" value={String(Object.keys(calendarData).length)} />
            <StatRow
              label="Win Rate by Day"
              value={`${Math.round((Object.values(calendarData).filter(v => v > 0).length / Object.keys(calendarData).length) * 100)}%`}
            />
            <StatRow label="Avg P&L per Day" value={`$${Math.round(Object.values(calendarData).reduce((a, b) => a + b, 0) / Object.keys(calendarData).length)}`} />
          </div>
        </ModalSection>

        <ModalSection title="Week-by-Week">
          {EMPTY_PNL_WEEKLY.map((week, i) => {
            const pct = (week.pnl / Math.max(...EMPTY_PNL_WEEKLY.map(w => Math.abs(w.pnl)))) * 100;
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

        <ModalSection title="vs Previous Month">
          <div className="grid grid-cols-2 gap-x-6">
            <StatRow label="Feb Total P&L" value={`$${EMPTY_MONTHLY_SUMMARY.prevMonthPnL.toLocaleString()}`} />
            <StatRow label="Feb Win Rate" value={`${EMPTY_MONTHLY_SUMMARY.prevMonthWinRate}%`} />
            <StatRow label="Feb Trading Days" value={String(EMPTY_MONTHLY_SUMMARY.prevMonthTradingDays)} />
            <StatRow
              label="Change"
              value={`+${Math.round(((Object.values(calendarData).reduce((a, b) => a + b, 0) - EMPTY_MONTHLY_SUMMARY.prevMonthPnL) / EMPTY_MONTHLY_SUMMARY.prevMonthPnL) * 100)}%`}
              color="var(--green)"
            />
          </div>
        </ModalSection>

        <ModalSection title="Consistency Score">
          <div className="flex items-center gap-3 mb-2">
            <span className="text-2xl font-bold" style={{ color: EMPTY_MONTHLY_SUMMARY.consistencyScore >= 60 ? 'var(--green)' : 'var(--orange)' }}>
              {EMPTY_MONTHLY_SUMMARY.consistencyScore}
            </span>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>/ 100</span>
          </div>
          <p className="text-[11px] leading-relaxed" style={{ color: 'var(--text-muted)' }}>
            {EMPTY_MONTHLY_SUMMARY.consistencyExplanation}
          </p>
        </ModalSection>

        <ModalSection title="Goal Progress">
          <GoalProgress current={EMPTY_MONTHLY_SUMMARY.goalProgress.current} target={EMPTY_MONTHLY_SUMMARY.goalProgress.target} />
        </ModalSection>
      </Modal>

      {/* 4. Position Detail Modal (per position) */}
      {showPositionDetail && (() => {
        const pos = positions.find(p => p.id === showPositionDetail);
        const detail = EMPTY_POSITION_DETAILS[showPositionDetail];
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
        const stratItem = healthStrategies.find(s => s.id === showHealthDetail);
        const portItem = healthPortfolios.find(p => p.id === showHealthDetail);
        const item = stratItem || portItem;
        const detail = EMPTY_HEALTH_DETAILS[showHealthDetail];
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
                  Expected: {isPortfolio ? `$${(portItem?.expectedPnL ?? 0).toLocaleString()}` : `${stratItem?.expectedR ?? 0}R`}
                </span>
                <span className="text-[10px] font-mono" style={{ color: sdColor(item.deviationSd) }}>
                  Actual: {isPortfolio ? `$${(portItem?.actualPnL ?? 0).toLocaleString()}` : `${stratItem?.actualR ?? 0}R`}
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
              data={EMPTY_VIX_HISTORY.map(v => v.value)}
              width={660}
              height={80}
              color={marketData.vix < 20 ? 'var(--green)' : 'var(--red)'}
            />
          </div>
          <div className="flex items-center justify-between mt-2">
            <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
              {EMPTY_VIX_HISTORY.length > 0 ? EMPTY_VIX_HISTORY[0].day : '--'}
            </span>
            <span className="text-[10px] font-mono" style={{ color: marketData.vix < 20 ? 'var(--green)' : 'var(--red)' }}>
              Current: {marketData.vix || '--'}
            </span>
            <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
              {EMPTY_VIX_HISTORY.length > 0 ? EMPTY_VIX_HISTORY[EMPTY_VIX_HISTORY.length - 1].day : '--'}
            </span>
          </div>
        </ModalSection>

        <ModalSection title="Market Breadth">
          <p className="text-xs text-center py-3" style={{ color: 'var(--text-muted)' }}>
            Market breadth data not available — requires premium data feed.
          </p>
        </ModalSection>

        <ModalSection title="Index Performance">
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)' }}>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold">SPY</span>
                <span className="text-xs font-mono" style={{ color: marketData.spy.changePct >= 0 ? 'var(--green)' : 'var(--red)' }}>
                  {marketData.spy.changePct >= 0 ? '+' : ''}{marketData.spy.changePct.toFixed(2)}%
                </span>
              </div>
              <MiniSparkline
                data={[0]} /* {{spy_intraday}} */
                width={280}
                height={40}
                color={marketData.spy.changePct >= 0 ? 'var(--green)' : 'var(--red)'}
              />
              <p className="text-xs font-mono mt-1" style={{ color: 'var(--text-primary)' }}>${marketData.spy.price}</p>
            </div>
            <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)' }}>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold">QQQ</span>
                <span className="text-xs font-mono" style={{ color: marketData.qqq.changePct >= 0 ? 'var(--green)' : 'var(--red)' }}>
                  {marketData.qqq.changePct >= 0 ? '+' : ''}{marketData.qqq.changePct.toFixed(2)}%
                </span>
              </div>
              <MiniSparkline
                data={[0]} /* {{qqq_intraday}} */
                width={280}
                height={40}
                color={marketData.qqq.changePct >= 0 ? 'var(--green)' : 'var(--red)'}
              />
              <p className="text-xs font-mono mt-1" style={{ color: 'var(--text-primary)' }}>${marketData.qqq.price}</p>
            </div>
          </div>
        </ModalSection>

        <ModalSection title="Regime History">
          <div className="space-y-0">
            {EMPTY_REGIME_HISTORY.map((entry, i) => {
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
            Regime impact analysis will be available when market data endpoints are built.
          </p>
        </ModalSection>
      </Modal>

      {/* 7. Goal Details Modal */}
      <Modal
        title="Goal Details"
        isOpen={showGoalDetail}
        onClose={() => setShowGoalDetail(false)}
        width="640px"
      >
        <ModalSection title="Progress Breakdown by Week">
          {EMPTY_GOAL_WEEKLY_PROGRESS.map((week, i) => (
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
            const now = new Date();
            const daysInMonth = new Date(now.getFullYear(), now.getMonth() + 1, 0).getDate();
            const daysRemaining = Math.max(daysInMonth - now.getDate(), 1);
            const remaining = monthlyTarget - monthlyProgress;
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
          <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
            Strategy contribution breakdown will be available when goal tracking endpoints are built.
          </p>
        </ModalSection>

        <ModalSection title="Historical Goal Achievement">
          <div className="space-y-0">
            {EMPTY_GOAL_HISTORY.map((month, i) => (
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
        <ModalSection title={`Active Issues (${issues.length})`}>
          <div className="space-y-2">
            {issues.map((issue) => {
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

        <ModalSection title={`Resolved Issues (${EMPTY_RESOLVED_ISSUES.length})`}>
          <div className="space-y-0">
            {EMPTY_RESOLVED_ISSUES.map((issue) => {
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
            Showing {filteredActivity.length} of {activityFeed.length} activities
          </p>
        </div>
      </Modal>

    </div>
  );
}
