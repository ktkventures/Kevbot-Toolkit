'use client';

import { useState, useMemo, useCallback } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import MetricCard from '@/components/MetricCard';
import Modal from '@/components/Modal';

// =============================================================================
// V4 Meta: Creative — Portfolio Command Center
// =============================================================================
// Visual-first portfolio management dashboard with:
//
// 1. STRATEGY CONTRIBUTION DONUT — animated donut chart of P&L per strategy
// 2. RISK ALLOCATION TREEMAP — visual treemap of capital allocation
// 3. DAILY P&L CALENDAR — heatmap calendar of trading days
// 4. DRAWDOWN WATERFALL — stepped chart showing DD accumulation/recovery
// 5. STRATEGY HEALTH GRID — dot matrix health indicators with tooltips
// 6. COMPLIANCE GAUGES — radial gauges per rule showing proximity to limits
// 7. PERFORMANCE VS BENCHMARK — overlay equity curves
// 8. SMART REBALANCING — AI-style suggestions for risk optimization
// =============================================================================

// ---------------------------------------------------------------------------
// CSS Animations
// ---------------------------------------------------------------------------

const ANIMATION_STYLES = `
@keyframes fade-in-up {
  0% { opacity: 0; transform: translateY(12px); }
  100% { opacity: 1; transform: translateY(0); }
}
@keyframes donut-draw {
  0% { stroke-dashoffset: 502; }
}
@keyframes pulse-ring {
  0% { transform: scale(1); opacity: 0.6; }
  100% { transform: scale(2.5); opacity: 0; }
}
@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
@keyframes gauge-fill {
  0% { stroke-dashoffset: 283; }
}
@keyframes glow-pulse {
  0%, 100% { box-shadow: 0 0 8px rgba(0,255,136,0.1); }
  50% { box-shadow: 0 0 20px rgba(0,255,136,0.3); }
}
@keyframes slide-in-right {
  0% { transform: translateX(16px); opacity: 0; }
  100% { transform: translateX(0); opacity: 1; }
}
@keyframes waterfall-grow {
  0% { transform: scaleY(0); }
  100% { transform: scaleY(1); }
}
@keyframes heartbeat {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.05); }
}
`;

// ---------------------------------------------------------------------------
// Mock Data
// ---------------------------------------------------------------------------

const PORTFOLIO = {
  name: 'Main Portfolio',
  strategies: 5,
  startingBalance: 80000,
  currentBalance: 84230,
  totalPnL: 4230,
  todayPnL: 247,
  totalTrades: 847,
  winRate: 55.2,
  profitFactor: 1.89,
  maxDrawdown: -2.1,
  sharpe: 1.42,
  avgR: 0.34,
};

const STRATEGIES = [
  {
    id: '1', name: 'SPY LONG - Mass #1', symbol: 'SPY', direction: 'LONG',
    pnl: 1680, pnlPct: 39.7, risk: 100, winRate: 58.3, pf: 2.12, trades: 312,
    allocation: 35, health: 'healthy' as const,
    healthNote: 'Consistent edge, low drawdown',
    equity: [0, 200, 350, 520, 680, 800, 920, 1050, 1180, 1350, 1420, 1550, 1680],
  },
  {
    id: '2', name: 'NVDA LONG - Mass #2', symbol: 'NVDA', direction: 'LONG',
    pnl: 1380, pnlPct: 32.6, risk: 100, winRate: 54.0, pf: 2.05, trades: 224,
    allocation: 28, health: 'warning' as const,
    healthNote: 'Win rate declining over last 20 trades',
    equity: [0, 150, 280, 200, 350, 500, 620, 710, 800, 950, 1100, 1250, 1380],
  },
  {
    id: '3', name: 'AAPL LONG - Mass #5', symbol: 'AAPL', direction: 'LONG',
    pnl: 770, pnlPct: 18.2, risk: 75, winRate: 52.8, pf: 1.78, trades: 186,
    allocation: 18, health: 'healthy' as const,
    healthNote: 'Stable performance',
    equity: [0, 80, 120, 200, 310, 400, 420, 480, 530, 600, 650, 720, 770],
  },
  {
    id: '4', name: 'TSLA LONG - Mass #7', symbol: 'TSLA', direction: 'LONG',
    pnl: 520, pnlPct: 12.3, risk: 80, winRate: 51.2, pf: 1.65, trades: 59,
    allocation: 12, health: 'healthy' as const,
    healthNote: 'Small sample, promising edge',
    equity: [0, 40, 90, 150, 200, 280, 320, 350, 400, 430, 470, 500, 520],
  },
  {
    id: '5', name: 'META LONG - Mass #13', symbol: 'META', direction: 'LONG',
    pnl: -120, pnlPct: -2.8, risk: 50, winRate: 45.5, pf: 1.22, trades: 66,
    allocation: 7, health: 'critical' as const,
    healthNote: 'Below edge threshold, 5-trade loss streak',
    equity: [0, 30, 60, 20, -10, -40, -80, -60, -30, -70, -100, -90, -120],
  },
];

const STRATEGY_COLORS = [
  'var(--accent)', 'var(--green)', 'var(--orange)', 'var(--blue)', 'var(--red)',
];

const COMBINED_EQUITY = [0, 500, 900, 1090, 1530, 1940, 2200, 2530, 2880, 3260, 3540, 3930, 4230];
const BENCHMARK_EQUITY = [0, 300, 550, 700, 900, 1100, 1250, 1400, 1600, 1800, 1950, 2100, 2300];

// Calendar data: day number -> daily P&L
const CALENDAR_DATA: Record<number, number> = {
  1: 142, 2: -67, 3: 210, 4: 55, 5: -120,
  6: 0, 7: 0,
  8: 180, 9: 95, 10: -42, 11: 310, 12: -85,
  13: 0, 14: 0,
  15: 165, 16: 220, 17: -30, 18: 145, 19: 280,
  20: 0, 21: 47,
};

// Drawdown waterfall steps
const DRAWDOWN_STEPS = [
  { label: 'Wk 1', dd: -0.8, recovered: true },
  { label: 'Wk 2', dd: -1.5, recovered: true },
  { label: 'Wk 3', dd: -0.4, recovered: true },
  { label: 'Wk 4', dd: -2.1, recovered: true },
  { label: 'Wk 5', dd: -1.3, recovered: true },
  { label: 'Wk 6', dd: -0.6, recovered: true },
  { label: 'Wk 7', dd: -0.3, recovered: true },
  { label: 'Now', dd: -0.1, recovered: false },
];

const COMPLIANCE_RULES = [
  { name: 'Max Daily Loss', threshold: '$5,000', currentValue: 1240, maxValue: 5000, passing: true, icon: 'shield' },
  { name: 'Max Total DD', threshold: '$10,000', currentValue: 2180, maxValue: 10000, passing: true, icon: 'trending-down' },
  { name: 'Min Trading Days', threshold: '10 days', currentValue: 7, maxValue: 10, passing: false, icon: 'calendar' },
  { name: 'Profit Target', threshold: '$10,000', currentValue: 4230, maxValue: 10000, passing: false, icon: 'target' },
  { name: 'Max Position Size', threshold: '2%', currentValue: 0.12, maxValue: 2, passing: true, icon: 'layers' },
  { name: 'Weekend Holding', threshold: 'Closed Fri', currentValue: 100, maxValue: 100, passing: true, icon: 'check' },
];

const REBALANCE_SUGGESTIONS = [
  {
    severity: 'warning' as const,
    title: 'Reduce META LONG exposure',
    description: 'META is below edge threshold with a 5-trade loss streak. Consider reducing risk from $50 to $25 per trade or pausing until win rate recovers above 48%.',
  },
  {
    severity: 'info' as const,
    title: 'NVDA allocation exceeds target',
    description: 'NVDA is at 28% allocation vs 25% target. Recent win rate decline (54% -> 51% over 20 trades) suggests trimming $100 -> $80 risk per trade.',
  },
  {
    severity: 'success' as const,
    title: 'SPY LONG performing above expectation',
    description: 'SPY has maintained 58.3% win rate across 312 trades. Consider increasing allocation from 35% to 40% given strong Sharpe ratio of 2.1.',
  },
];

// ---------------------------------------------------------------------------
// SVG Components
// ---------------------------------------------------------------------------

/** Donut chart showing P&L contribution per strategy */
function StrategyDonut({ strategies, size = 200 }: { strategies: typeof STRATEGIES; size?: number }) {
  const positiveStrategies = strategies.filter((s) => s.pnl > 0);
  const totalPositivePnL = positiveStrategies.reduce((a, s) => a + s.pnl, 0);
  const r = (size - 20) / 2;
  const c = size / 2;
  const circumference = 2 * Math.PI * r;

  let offset = 0;
  const arcs = strategies.map((s, i) => {
    const pct = Math.abs(s.pnl) / (totalPositivePnL || 1);
    const len = pct * circumference * 0.95; // small gap
    const arc = { ...s, offset, len, color: STRATEGY_COLORS[i] };
    offset += len + circumference * 0.01; // gap
    return arc;
  });

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ transform: 'rotate(-90deg)' }}>
        {/* Background ring */}
        <circle cx={c} cy={c} r={r} fill="none" stroke="var(--bg-input)" strokeWidth="16" />
        {arcs.map((arc) => (
          <circle
            key={arc.id}
            cx={c}
            cy={c}
            r={r}
            fill="none"
            stroke={arc.pnl >= 0 ? arc.color : 'var(--red)'}
            strokeWidth={arc.pnl >= 0 ? 16 : 10}
            strokeDasharray={`${arc.len} ${circumference - arc.len}`}
            strokeDashoffset={-arc.offset}
            strokeLinecap="round"
            opacity={arc.pnl >= 0 ? 0.85 : 0.5}
            style={{ animation: 'donut-draw 1.5s ease-out forwards', transition: 'all 0.5s ease' }}
          />
        ))}
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-2xl font-bold" style={{ color: 'var(--green)' }}>
          +${PORTFOLIO.totalPnL.toLocaleString()}
        </span>
        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Total P&L</span>
      </div>
    </div>
  );
}

/** Risk allocation treemap */
function AllocationTreemap({ strategies }: { strategies: typeof STRATEGIES }) {
  const totalAlloc = strategies.reduce((a, s) => a + s.allocation, 0);
  // Simple horizontal treemap
  return (
    <div className="flex rounded-xl overflow-hidden" style={{ height: 80 }}>
      {strategies.map((s, i) => {
        const widthPct = (s.allocation / totalAlloc) * 100;
        const healthColors = {
          healthy: 'var(--green)',
          warning: 'var(--orange)',
          critical: 'var(--red)',
        };
        return (
          <div
            key={s.id}
            className="relative flex items-center justify-center overflow-hidden transition-all group"
            style={{
              width: `${widthPct}%`,
              background: STRATEGY_COLORS[i],
              opacity: 0.75,
              borderRight: i < strategies.length - 1 ? '2px solid var(--bg-primary)' : 'none',
              animation: `fade-in-up 0.6s ease-out ${i * 0.1}s both`,
            }}
            title={`${s.name}: ${s.allocation}% allocation`}
          >
            {widthPct > 12 && (
              <div className="text-center">
                <p className="text-xs font-bold text-white drop-shadow-sm">{s.symbol}</p>
                <p className="text-[10px] text-white/70">{s.allocation}%</p>
              </div>
            )}
            {/* Health indicator dot */}
            <div
              className="absolute bottom-1.5 right-1.5 w-2 h-2 rounded-full"
              style={{ background: healthColors[s.health], boxShadow: `0 0 6px ${healthColors[s.health]}` }}
            />
          </div>
        );
      })}
    </div>
  );
}

/** Mini SVG equity curve */
function MiniEquityCurve({ data, width = 320, height = 80, gradientId, color }: {
  data: number[];
  width?: number;
  height?: number;
  gradientId: string;
  color: string;
}) {
  if (data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const pad = 4;
  const chartH = height - pad * 2;
  const chartW = width - pad * 2;

  const points = data.map((v, i) => {
    const x = pad + (i / (data.length - 1)) * chartW;
    const y = pad + chartH - ((v - min) / range) * chartH;
    return `${x},${y}`;
  });

  const pathD = `M${points.join(' L')}`;
  const fillD = `${pathD} L${pad + chartW},${height} L${pad},${height} Z`;

  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity={0.25} />
          <stop offset="100%" stopColor={color} stopOpacity={0} />
        </linearGradient>
      </defs>
      <path d={fillD} fill={`url(#${gradientId})`} />
      <path d={pathD} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" />
    </svg>
  );
}

/** Calendar heatmap */
function PnLCalendar({ data }: { data: Record<number, number> }) {
  const today = 21;
  const daysInMonth = 31;
  const firstDayOffset = 5; // March 2026 starts on Sunday, 0-index
  const dayNames = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

  const cells: { day: number; pnl: number | null }[] = [];
  // Pad start
  for (let i = 0; i < firstDayOffset; i++) cells.push({ day: 0, pnl: null });
  for (let d = 1; d <= daysInMonth; d++) {
    cells.push({ day: d, pnl: data[d] ?? null });
  }

  const maxAbs = Math.max(
    ...Object.values(data).filter(Boolean).map(Math.abs),
    1
  );

  function getCellColor(pnl: number | null) {
    if (pnl === null || pnl === undefined) return 'transparent';
    if (pnl === 0) return 'var(--bg-input)';
    const intensity = Math.min(Math.abs(pnl) / maxAbs, 1);
    if (pnl > 0) return `rgba(0, 255, 136, ${0.15 + intensity * 0.55})`;
    return `rgba(255, 68, 68, ${0.15 + intensity * 0.55})`;
  }

  return (
    <div>
      {/* Day headers */}
      <div className="grid grid-cols-7 gap-1 mb-1">
        {dayNames.map((d) => (
          <div key={d} className="text-center text-[10px] font-medium" style={{ color: 'var(--text-muted)' }}>{d}</div>
        ))}
      </div>
      {/* Calendar grid */}
      <div className="grid grid-cols-7 gap-1">
        {cells.map((cell, i) => (
          <div
            key={i}
            className="aspect-square rounded-md flex flex-col items-center justify-center relative transition-all"
            style={{
              background: cell.day === 0 ? 'transparent' : getCellColor(cell.pnl),
              border: cell.day === today ? '2px solid var(--accent)' : cell.day === 0 ? 'none' : '1px solid var(--border)',
              animation: cell.day > 0 ? `fade-in-up 0.4s ease-out ${cell.day * 0.02}s both` : 'none',
              cursor: cell.pnl ? 'pointer' : 'default',
            }}
            title={cell.pnl ? `Day ${cell.day}: ${cell.pnl >= 0 ? '+' : ''}$${cell.pnl}` : undefined}
          >
            {cell.day > 0 && (
              <>
                <span className="text-[10px]" style={{ color: cell.day <= today ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                  {cell.day}
                </span>
                {cell.pnl !== null && cell.pnl !== 0 && (
                  <span
                    className="text-[8px] font-mono font-medium"
                    style={{ color: cell.pnl > 0 ? 'var(--green)' : 'var(--red)' }}
                  >
                    {cell.pnl > 0 ? '+' : ''}{cell.pnl}
                  </span>
                )}
              </>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

/** Drawdown waterfall chart */
function DrawdownWaterfall({ steps }: { steps: typeof DRAWDOWN_STEPS }) {
  const maxDD = Math.max(...steps.map((s) => Math.abs(s.dd)));
  const barWidth = 100 / steps.length;

  return (
    <div className="flex items-end justify-between" style={{ height: 140 }}>
      {steps.map((step, i) => {
        const heightPct = (Math.abs(step.dd) / maxDD) * 100;
        return (
          <div
            key={i}
            className="flex flex-col items-center gap-1"
            style={{ width: `${barWidth}%`, animation: `fade-in-up 0.5s ease-out ${i * 0.08}s both` }}
          >
            <span
              className="text-[10px] font-mono font-semibold"
              style={{ color: step.recovered ? 'var(--text-muted)' : 'var(--red)' }}
            >
              {step.dd}%
            </span>
            <div
              className="w-3/4 rounded-t-md transition-all"
              style={{
                height: `${heightPct}%`,
                minHeight: 8,
                background: step.recovered
                  ? `linear-gradient(180deg, var(--orange) 0%, var(--green) 100%)`
                  : 'var(--red)',
                opacity: step.recovered ? 0.6 : 0.9,
                animation: `waterfall-grow 0.8s ease-out ${i * 0.1}s both`,
                transformOrigin: 'bottom',
                boxShadow: !step.recovered ? '0 0 10px rgba(255,68,68,0.3)' : 'none',
              }}
            />
            <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{step.label}</span>
          </div>
        );
      })}
    </div>
  );
}

/** Compliance gauge (radial) */
function ComplianceGauge({ name, currentValue, maxValue, passing }: {
  name: string;
  currentValue: number;
  maxValue: number;
  passing: boolean;
}) {
  const pct = Math.min((currentValue / maxValue) * 100, 100);
  const r = 40;
  const circumference = 2 * Math.PI * r;
  const arcLength = (pct / 100) * circumference * 0.75; // 270 deg arc
  const bgArcLength = circumference * 0.75;

  const color = !passing
    ? 'var(--red)'
    : pct > 75
      ? 'var(--orange)'
      : 'var(--green)';

  return (
    <div className="flex flex-col items-center">
      <div className="relative" style={{ width: 96, height: 72 }}>
        <svg width={96} height={72} viewBox="0 0 96 90">
          {/* Background arc */}
          <circle
            cx={48}
            cy={48}
            r={r}
            fill="none"
            stroke="var(--bg-input)"
            strokeWidth={6}
            strokeDasharray={`${bgArcLength} ${circumference}`}
            strokeLinecap="round"
            transform="rotate(135 48 48)"
          />
          {/* Value arc */}
          <circle
            cx={48}
            cy={48}
            r={r}
            fill="none"
            stroke={color}
            strokeWidth={6}
            strokeDasharray={`${arcLength} ${circumference}`}
            strokeLinecap="round"
            transform="rotate(135 48 48)"
            style={{ animation: 'gauge-fill 1.2s ease-out forwards', filter: `drop-shadow(0 0 4px ${color})` }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center" style={{ top: 20 }}>
          <span className="text-lg font-bold font-mono" style={{ color }}>{pct.toFixed(0)}%</span>
        </div>
      </div>
      <span className="text-[10px] text-center mt-1" style={{ color: 'var(--text-muted)' }}>{name}</span>
      <span
        className="text-[9px] px-1.5 py-0.5 rounded-full mt-0.5 font-medium"
        style={{
          background: passing ? 'var(--green-muted)' : 'var(--red-muted)',
          color: passing ? 'var(--green)' : 'var(--red)',
        }}
      >
        {passing ? 'PASS' : 'FAIL'}
      </span>
    </div>
  );
}

/** Strategy health dot grid */
function HealthGrid({ strategies }: { strategies: typeof STRATEGIES }) {
  const healthColors = {
    healthy: 'var(--green)',
    warning: 'var(--orange)',
    critical: 'var(--red)',
  };

  return (
    <div className="flex items-center gap-4">
      {strategies.map((s, i) => (
        <div
          key={s.id}
          className="flex flex-col items-center gap-1.5 group cursor-pointer"
          style={{ animation: `fade-in-up 0.4s ease-out ${i * 0.08}s both` }}
          title={`${s.name}: ${s.healthNote}`}
        >
          <div className="relative">
            <div
              className="w-5 h-5 rounded-full transition-all"
              style={{
                background: healthColors[s.health],
                boxShadow: `0 0 8px ${healthColors[s.health]}`,
              }}
            />
            {s.health === 'critical' && (
              <div
                className="absolute inset-0 w-5 h-5 rounded-full"
                style={{
                  background: healthColors[s.health],
                  animation: 'pulse-ring 2s ease-out infinite',
                }}
              />
            )}
          </div>
          <span className="text-[10px] font-medium" style={{ color: 'var(--text-muted)' }}>{s.symbol}</span>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Strategy Detail Modal
// ---------------------------------------------------------------------------

function StrategyDetailModal({ strategy, index, onClose }: {
  strategy: typeof STRATEGIES[0];
  index: number;
  onClose: () => void;
}) {
  const color = STRATEGY_COLORS[index];

  return (
    <Modal title={strategy.name} isOpen onClose={onClose} width="700px">
      <div className="space-y-6">
        {/* Header stats */}
        <div className="grid grid-cols-5 gap-3">
          <div className="text-center p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>P&L</p>
            <p className="text-lg font-bold" style={{ color: strategy.pnl >= 0 ? 'var(--green)' : 'var(--red)' }}>
              {strategy.pnl >= 0 ? '+' : ''}${strategy.pnl.toLocaleString()}
            </p>
          </div>
          <div className="text-center p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Win Rate</p>
            <p className="text-lg font-bold">{strategy.winRate}%</p>
          </div>
          <div className="text-center p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>PF</p>
            <p className="text-lg font-bold">{strategy.pf}</p>
          </div>
          <div className="text-center p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Trades</p>
            <p className="text-lg font-bold">{strategy.trades}</p>
          </div>
          <div className="text-center p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Allocation</p>
            <p className="text-lg font-bold">{strategy.allocation}%</p>
          </div>
        </div>

        {/* Equity curve */}
        <div>
          <h4 className="text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Equity Curve</h4>
          <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)' }}>
            <MiniEquityCurve
              data={strategy.equity}
              height={200}
              gradientId={`modal-eq-${strategy.id}`}
              color={color}
            />
          </div>
        </div>

        {/* Health status */}
        <div
          className="rounded-lg p-4 flex items-center gap-3"
          style={{
            background: 'var(--bg-input)',
            borderLeft: `3px solid ${strategy.health === 'healthy' ? 'var(--green)' : strategy.health === 'warning' ? 'var(--orange)' : 'var(--red)'}`,
          }}
        >
          <div
            className="w-3 h-3 rounded-full shrink-0"
            style={{
              background: strategy.health === 'healthy' ? 'var(--green)' : strategy.health === 'warning' ? 'var(--orange)' : 'var(--red)',
              boxShadow: `0 0 6px ${strategy.health === 'healthy' ? 'var(--green)' : strategy.health === 'warning' ? 'var(--orange)' : 'var(--red)'}`,
            }}
          />
          <div>
            <p className="text-sm font-medium capitalize">{strategy.health}</p>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{strategy.healthNote}</p>
          </div>
        </div>
      </div>
    </Modal>
  );
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function PortfolioDetailV4() {
  const [selectedStrategy, setSelectedStrategy] = useState<number | null>(null);
  const [showBenchmark, setShowBenchmark] = useState(true);

  return (
    <div>
      <style dangerouslySetInnerHTML={{ __html: ANIMATION_STYLES }} />

      <PageHeader
        title={PORTFOLIO.name}
        subtitle={`${PORTFOLIO.strategies} strategies | $${PORTFOLIO.startingBalance.toLocaleString()} starting balance`}
        backHref="/portfolios"
        actions={
          <>
            <button
              className="px-3 py-1.5 rounded-lg text-sm font-medium transition-all"
              style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
            >
              Edit
            </button>
            <button
              className="px-3 py-1.5 rounded-lg text-sm font-medium transition-all"
              style={{ background: 'var(--accent)', color: 'white' }}
            >
              Deploy
            </button>
          </>
        }
      />

      {/* ── Row 1: KPI Strip ── */}
      <div className="grid grid-cols-6 gap-3 mb-6">
        {[
          { label: 'Balance', value: `$${PORTFOLIO.currentBalance.toLocaleString()}`, delta: `+$${PORTFOLIO.todayPnL}`, positive: true },
          { label: 'Total P&L', value: `+$${PORTFOLIO.totalPnL.toLocaleString()}`, delta: `+${((PORTFOLIO.totalPnL / PORTFOLIO.startingBalance) * 100).toFixed(1)}%`, positive: true },
          { label: 'Win Rate', value: `${PORTFOLIO.winRate}%` },
          { label: 'Profit Factor', value: `${PORTFOLIO.profitFactor}` },
          { label: 'Sharpe', value: `${PORTFOLIO.sharpe}` },
          { label: 'Max DD', value: `${PORTFOLIO.maxDrawdown}%` },
        ].map((m, i) => (
          <div
            key={m.label}
            style={{ animation: `fade-in-up 0.4s ease-out ${i * 0.06}s both` }}
          >
            <MetricCard {...m} />
          </div>
        ))}
      </div>

      {/* ── Row 2: Donut + Allocation + Health ── */}
      <div className="grid grid-cols-3 gap-5 mb-6">
        {/* Strategy Contribution Donut */}
        <Card>
          <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
            P&L Contribution
          </h3>
          <div className="flex justify-center mb-4">
            <StrategyDonut strategies={STRATEGIES} size={180} />
          </div>
          {/* Legend */}
          <div className="space-y-1.5">
            {STRATEGIES.map((s, i) => (
              <button
                key={s.id}
                className="flex items-center justify-between w-full px-2 py-1 rounded-md transition-all"
                style={{ background: 'transparent' }}
                onClick={() => setSelectedStrategy(i)}
                onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-input)')}
                onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
              >
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full" style={{ background: STRATEGY_COLORS[i] }} />
                  <span className="text-xs">{s.symbol}</span>
                </div>
                <span
                  className="text-xs font-mono font-medium"
                  style={{ color: s.pnl >= 0 ? 'var(--green)' : 'var(--red)' }}
                >
                  {s.pnl >= 0 ? '+' : ''}${s.pnl.toLocaleString()} ({s.pnlPct}%)
                </span>
              </button>
            ))}
          </div>
        </Card>

        {/* Risk Allocation + Health Grid */}
        <Card>
          <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
            Risk Allocation
          </h3>
          <AllocationTreemap strategies={STRATEGIES} />

          <div className="mt-6">
            <h4 className="text-xs font-medium mb-3" style={{ color: 'var(--text-muted)' }}>Strategy Health</h4>
            <HealthGrid strategies={STRATEGIES} />
          </div>

          {/* Allocation table */}
          <div className="mt-5 space-y-1.5">
            {STRATEGIES.map((s, i) => (
              <div key={s.id} className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-sm" style={{ background: STRATEGY_COLORS[i] }} />
                  <span>{s.symbol}</span>
                </div>
                <div className="flex items-center gap-3">
                  <span style={{ color: 'var(--text-muted)' }}>${s.risk}/trade</span>
                  <span className="font-mono font-medium w-8 text-right">{s.allocation}%</span>
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* P&L Calendar */}
        <Card>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
              March 2026
            </h3>
            <div className="flex items-center gap-2">
              <span className="flex items-center gap-1 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                <span className="w-2.5 h-2.5 rounded-sm" style={{ background: 'rgba(0,255,136,0.5)' }} />
                Profit
              </span>
              <span className="flex items-center gap-1 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                <span className="w-2.5 h-2.5 rounded-sm" style={{ background: 'rgba(255,68,68,0.5)' }} />
                Loss
              </span>
            </div>
          </div>
          <PnLCalendar data={CALENDAR_DATA} />
        </Card>
      </div>

      {/* ── Row 3: Equity vs Benchmark + Drawdown Waterfall ── */}
      <div className="grid grid-cols-2 gap-5 mb-6">
        {/* Performance vs Benchmark */}
        <Card>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
              Performance vs Benchmark
            </h3>
            <label className="flex items-center gap-2 cursor-pointer">
              <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Show SPY</span>
              <button
                onClick={() => setShowBenchmark(!showBenchmark)}
                className="relative rounded-full transition-colors"
                style={{
                  width: 28,
                  height: 14,
                  background: showBenchmark ? 'var(--accent)' : 'var(--bg-input)',
                  border: showBenchmark ? 'none' : '1px solid var(--border)',
                }}
              >
                <span
                  className="absolute rounded-full transition-all"
                  style={{
                    width: 10,
                    height: 10,
                    top: 2,
                    left: showBenchmark ? 16 : 2,
                    background: showBenchmark ? 'white' : 'var(--text-muted)',
                  }}
                />
              </button>
            </label>
          </div>
          <div className="rounded-lg p-3 relative" style={{ background: 'var(--bg-input)' }}>
            {showBenchmark && (
              <div style={{ position: 'absolute', top: 12, left: 12, right: 12, opacity: 0.4 }}>
                <MiniEquityCurve
                  data={BENCHMARK_EQUITY}
                  height={220}
                  gradientId="v4-benchmark"
                  color="var(--text-muted)"
                />
              </div>
            )}
            <div style={{ position: 'relative' }}>
              <MiniEquityCurve
                data={COMBINED_EQUITY}
                height={220}
                gradientId="v4-portfolio"
                color="var(--green)"
              />
            </div>
          </div>
          <div className="flex items-center gap-6 mt-3">
            <span className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--text-muted)' }}>
              <span className="w-3 h-0.5 inline-block" style={{ background: 'var(--green)' }} />
              Portfolio (+{((PORTFOLIO.totalPnL / PORTFOLIO.startingBalance) * 100).toFixed(1)}%)
            </span>
            {showBenchmark && (
              <span className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--text-muted)' }}>
                <span className="w-3 h-0.5 inline-block" style={{ background: 'var(--text-muted)', opacity: 0.5 }} />
                SPY Benchmark (+2.9%)
              </span>
            )}
          </div>
        </Card>

        {/* Drawdown Waterfall */}
        <Card>
          <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
            Drawdown Waterfall
          </h3>
          <DrawdownWaterfall steps={DRAWDOWN_STEPS} />
          <div className="grid grid-cols-4 gap-3 mt-5">
            <div>
              <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Max DD</p>
              <p className="text-sm font-bold" style={{ color: 'var(--red)' }}>-2.1%</p>
            </div>
            <div>
              <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Avg DD</p>
              <p className="text-sm font-bold">-0.89%</p>
            </div>
            <div>
              <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Avg Recovery</p>
              <p className="text-sm font-bold">3.2 days</p>
            </div>
            <div>
              <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Current</p>
              <p className="text-sm font-bold" style={{ color: 'var(--green)' }}>-0.1%</p>
            </div>
          </div>
        </Card>
      </div>

      {/* ── Row 4: Compliance Gauges ── */}
      <Card className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
            Compliance Dashboard
          </h3>
          <span
            className="text-xs px-2.5 py-1 rounded-full font-medium"
            style={{
              background: COMPLIANCE_RULES.every((r) => r.passing) ? 'var(--green-muted)' : 'var(--red-muted)',
              color: COMPLIANCE_RULES.every((r) => r.passing) ? 'var(--green)' : 'var(--red)',
            }}
          >
            {COMPLIANCE_RULES.filter((r) => r.passing).length}/{COMPLIANCE_RULES.length} Passing
          </span>
        </div>
        <div className="grid grid-cols-6 gap-4">
          {COMPLIANCE_RULES.map((rule) => (
            <ComplianceGauge
              key={rule.name}
              name={rule.name}
              currentValue={rule.currentValue}
              maxValue={rule.maxValue}
              passing={rule.passing}
            />
          ))}
        </div>
      </Card>

      {/* ── Row 5: Smart Rebalancing Suggestions ── */}
      <Card>
        <div className="flex items-center gap-2 mb-4">
          <div
            className="w-6 h-6 rounded-lg flex items-center justify-center"
            style={{ background: 'var(--accent-muted)' }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round">
              <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
            </svg>
          </div>
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
            Smart Rebalancing Suggestions
          </h3>
        </div>
        <div className="space-y-3">
          {REBALANCE_SUGGESTIONS.map((suggestion, i) => {
            const colors = {
              warning: { border: 'var(--orange)', bg: 'rgba(255,170,0,0.06)', icon: 'var(--orange)' },
              info: { border: 'var(--blue)', bg: 'rgba(59,130,246,0.06)', icon: 'var(--blue)' },
              success: { border: 'var(--green)', bg: 'rgba(0,255,136,0.06)', icon: 'var(--green)' },
            };
            const c = colors[suggestion.severity];

            return (
              <div
                key={i}
                className="rounded-lg p-4 transition-all"
                style={{
                  background: c.bg,
                  borderLeft: `3px solid ${c.border}`,
                  animation: `slide-in-right 0.5s ease-out ${i * 0.12}s both`,
                }}
              >
                <div className="flex items-start gap-3">
                  <div
                    className="w-2 h-2 rounded-full mt-1.5 shrink-0"
                    style={{ background: c.icon, boxShadow: `0 0 6px ${c.icon}` }}
                  />
                  <div>
                    <p className="text-sm font-medium mb-1">{suggestion.title}</p>
                    <p className="text-xs leading-relaxed" style={{ color: 'var(--text-muted)' }}>
                      {suggestion.description}
                    </p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </Card>

      {/* Strategy Detail Modal */}
      {selectedStrategy !== null && (
        <StrategyDetailModal
          strategy={STRATEGIES[selectedStrategy]}
          index={selectedStrategy}
          onClose={() => setSelectedStrategy(null)}
        />
      )}
    </div>
  );
}
