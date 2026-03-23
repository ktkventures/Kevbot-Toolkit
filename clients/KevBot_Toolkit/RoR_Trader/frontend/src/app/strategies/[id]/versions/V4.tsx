'use client';

import { useState, useMemo } from 'react';
import PageHeader from '@/components/PageHeader';
import Card from '@/components/Card';
import MetricCard from '@/components/MetricCard';

// =============================================================================
// V4 Meta: Creative — Strategy Health Dashboard + Visual Analytics
// =============================================================================
// The "wow factor" strategy detail. Key creative features:
//
// 1. HEALTH DASHBOARD — circular health score (0-100) with segmented ring
//    for win rate health, drawdown health, consistency health, risk control
// 2. INTERACTIVE TRADE TIMELINE — horizontal timeline of trades with
//    color-coded entry/exit points, hover details in tooltip
// 3. PERFORMANCE HEATMAP — day-of-week x hour-of-day grid showing when
//    the strategy performs best (green) and worst (red)
// 4. EQUITY CURVE WITH REGIME OVERLAY — equity line with background bands
//    showing bull/bear/neutral market regimes
// 5. STRATEGY DNA CARD — radar chart showing strategy characteristics
// 6. LIVE POSITION WIDGET — animated card with pulsing dot for active status
// 7. CONFLUENCE EFFECTIVENESS — horizontal bars showing which conditions
//    contribute most to wins vs losses
// 8. RISK/REWARD SCATTER — each trade as a colored dot (x=risk, y=reward)
// 9. SMART SUMMARY — AI-style text insight about the strategy
// =============================================================================

// ---------------------------------------------------------------------------
// CSS Animations
// ---------------------------------------------------------------------------

const ANIMATION_STYLES = `
@keyframes health-ring-fill {
  0% { stroke-dasharray: 0 314; }
}
@keyframes fade-in-up {
  0% { opacity: 0; transform: translateY(8px); }
  100% { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-live {
  0%, 100% { opacity: 0.4; transform: scale(1); }
  50% { opacity: 1; transform: scale(1.6); }
}
@keyframes glow-pulse {
  0%, 100% { box-shadow: 0 0 6px rgba(0,255,136,0.1); }
  50% { box-shadow: 0 0 18px rgba(0,255,136,0.3); }
}
@keyframes radar-grow {
  0% { transform: scale(0); opacity: 0; }
  100% { transform: scale(1); opacity: 1; }
}
@keyframes typing-cursor {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}
@keyframes timeline-draw {
  0% { stroke-dashoffset: 2000; }
  100% { stroke-dashoffset: 0; }
}
@keyframes scatter-pop {
  0% { opacity: 0; r: 0; }
  100% { opacity: 1; }
}
@keyframes regime-fade {
  0% { opacity: 0; }
  100% { opacity: 1; }
}
`;

// ---------------------------------------------------------------------------
// Mock Data
// ---------------------------------------------------------------------------

const strategy = {
  id: '1',
  name: 'NVDA LONG - Mass #2',
  symbol: 'NVDA',
  direction: 'LONG' as const,
  timeframe: '1Min',
  session: 'RTH',
  trigger: '[C] EMA Bull Cross',
  exitTrigger: '4-bar exit',
  stop: 'Swing (5 bars, $0.03 pad)',
  target: '2R',
  confluence: ['5M-RVOL-HIGH', '1D-MACD_LINE-BULL'],
  generalConfluence: ['GEN-TIME_OF_DAY-MORNING'],
  forwardTesting: true,
  forwardTestStart: '2026-03-07',
  alertTracking: true,
  monitored: true,
  btDays: 90,
  btStart: '2025-12-07',
  btEnd: '2026-03-07',
};

const kpis = {
  trades: 224, winRate: 54.0, pf: 2.05, avgR: 0.42,
  totalR: 94.1, dailyR: 1.95, rSquared: 0.91, maxDD: -4.5,
  avgWinR: 1.85, avgLossR: -0.90,
};

const extKpis = {
  wins: 121, losses: 103,
  bestTrade: 5.23, worstTrade: -2.10,
  sharpe: 1.82, sortino: 2.65, calmar: 4.12, kelly: 0.182,
  recoveryFactor: 20.9, maxConsecWins: 8, maxConsecLosses: 5,
  avgTradeDuration: '4m 23s',
};

const fwdKpis = {
  trades: 18, winRate: 52.1, pf: 1.82, avgR: 0.35,
  totalR: 6.3, dailyR: 0.45,
};

// Trades with entry/exit for timeline + scatter
const mockTrades = [
  { id: 1, entryTime: '09:42', exitTime: '09:46', date: '03/18', entryPrice: 142.35, exitPrice: 142.98, pnlR: 1.42, execType: 'C', exitReason: 'Signal', risk: 0.63, reward: 1.42 },
  { id: 2, entryTime: '10:15', exitTime: '10:19', date: '03/18', entryPrice: 143.10, exitPrice: 142.65, pnlR: -0.85, execType: 'C', exitReason: 'Stop', risk: 0.45, reward: -0.85 },
  { id: 3, entryTime: '11:02', exitTime: '11:06', date: '03/18', entryPrice: 143.50, exitPrice: 144.12, pnlR: 2.07, execType: 'L0', exitReason: 'Target', risk: 0.62, reward: 2.07 },
  { id: 4, entryTime: '09:35', exitTime: '09:39', date: '03/17', entryPrice: 141.80, exitPrice: 141.45, pnlR: -0.92, execType: 'C', exitReason: 'Stop', risk: 0.35, reward: -0.92 },
  { id: 5, entryTime: '10:22', exitTime: '10:26', date: '03/17', entryPrice: 141.60, exitPrice: 142.25, pnlR: 1.63, execType: 'HM', exitReason: 'Signal', risk: 0.65, reward: 1.63 },
  { id: 6, entryTime: '11:45', exitTime: '11:49', date: '03/17', entryPrice: 142.90, exitPrice: 143.55, pnlR: 1.95, execType: 'C', exitReason: 'Target', risk: 0.65, reward: 1.95 },
  { id: 7, entryTime: '13:10', exitTime: '13:14', date: '03/17', entryPrice: 143.20, exitPrice: 142.80, pnlR: -0.67, execType: 'L0', exitReason: 'Bar Count', risk: 0.40, reward: -0.67 },
  { id: 8, entryTime: '09:48', exitTime: '09:52', date: '03/14', entryPrice: 140.50, exitPrice: 141.30, pnlR: 2.38, execType: 'C', exitReason: 'Signal', risk: 0.80, reward: 2.38 },
  { id: 9, entryTime: '10:30', exitTime: '10:34', date: '03/14', entryPrice: 141.10, exitPrice: 140.78, pnlR: -0.53, execType: 'HL', exitReason: 'Stop', risk: 0.32, reward: -0.53 },
  { id: 10, entryTime: '11:15', exitTime: '11:19', date: '03/14', entryPrice: 140.95, exitPrice: 141.72, pnlR: 1.78, execType: 'C', exitReason: 'Target', risk: 0.77, reward: 1.78 },
  { id: 11, entryTime: '09:32', exitTime: '09:36', date: '03/13', entryPrice: 139.80, exitPrice: 140.45, pnlR: 1.15, execType: 'C', exitReason: 'Signal', risk: 0.65, reward: 1.15 },
  { id: 12, entryTime: '10:55', exitTime: '10:59', date: '03/13', entryPrice: 140.30, exitPrice: 139.90, pnlR: -0.78, execType: 'L0', exitReason: 'Stop', risk: 0.40, reward: -0.78 },
  { id: 13, entryTime: '09:38', exitTime: '09:44', date: '03/12', entryPrice: 138.90, exitPrice: 139.65, pnlR: 1.50, execType: 'C', exitReason: 'Signal', risk: 0.75, reward: 1.50 },
  { id: 14, entryTime: '10:45', exitTime: '10:50', date: '03/12', entryPrice: 139.40, exitPrice: 139.10, pnlR: -0.60, execType: 'C', exitReason: 'Stop', risk: 0.30, reward: -0.60 },
  { id: 15, entryTime: '13:20', exitTime: '13:28', date: '03/12', entryPrice: 139.55, exitPrice: 140.80, pnlR: 2.50, execType: 'C', exitReason: 'Target', risk: 1.25, reward: 2.50 },
  { id: 16, entryTime: '09:31', exitTime: '09:35', date: '03/11', entryPrice: 137.40, exitPrice: 138.05, pnlR: 1.30, execType: 'HM', exitReason: 'Signal', risk: 0.65, reward: 1.30 },
];

// Performance heatmap: DoW x ToD
const heatmapData: { day: string; hour: string; avgR: number; trades: number }[] = [
  // Monday
  { day: 'Mon', hour: '09:30', avgR: 0.42, trades: 8 },
  { day: 'Mon', hour: '10:00', avgR: 0.18, trades: 6 },
  { day: 'Mon', hour: '11:00', avgR: 0.31, trades: 4 },
  { day: 'Mon', hour: '13:00', avgR: -0.05, trades: 3 },
  { day: 'Mon', hour: '14:00', avgR: -0.22, trades: 3 },
  { day: 'Mon', hour: '15:00', avgR: 0.15, trades: 2 },
  // Tuesday
  { day: 'Tue', hour: '09:30', avgR: 0.65, trades: 9 },
  { day: 'Tue', hour: '10:00', avgR: 0.38, trades: 7 },
  { day: 'Tue', hour: '11:00', avgR: 0.22, trades: 5 },
  { day: 'Tue', hour: '13:00', avgR: 0.41, trades: 4 },
  { day: 'Tue', hour: '14:00', avgR: 0.10, trades: 2 },
  { day: 'Tue', hour: '15:00', avgR: 0.55, trades: 2 },
  // Wednesday
  { day: 'Wed', hour: '09:30', avgR: 0.55, trades: 8 },
  { day: 'Wed', hour: '10:00', avgR: 0.25, trades: 6 },
  { day: 'Wed', hour: '11:00', avgR: 0.35, trades: 5 },
  { day: 'Wed', hour: '13:00', avgR: 0.28, trades: 4 },
  { day: 'Wed', hour: '14:00', avgR: -0.12, trades: 3 },
  { day: 'Wed', hour: '15:00', avgR: 0.42, trades: 2 },
  // Thursday
  { day: 'Thu', hour: '09:30', avgR: 0.30, trades: 7 },
  { day: 'Thu', hour: '10:00', avgR: -0.08, trades: 5 },
  { day: 'Thu', hour: '11:00', avgR: 0.15, trades: 4 },
  { day: 'Thu', hour: '13:00', avgR: -0.18, trades: 3 },
  { day: 'Thu', hour: '14:00', avgR: -0.35, trades: 3 },
  { day: 'Thu', hour: '15:00', avgR: 0.08, trades: 1 },
  // Friday
  { day: 'Fri', hour: '09:30', avgR: 0.48, trades: 6 },
  { day: 'Fri', hour: '10:00', avgR: 0.20, trades: 5 },
  { day: 'Fri', hour: '11:00', avgR: 0.28, trades: 3 },
  { day: 'Fri', hour: '13:00', avgR: 0.12, trades: 2 },
  { day: 'Fri', hour: '14:00', avgR: -0.05, trades: 2 },
  { day: 'Fri', hour: '15:00', avgR: 0.35, trades: 1 },
];

// Confluence effectiveness
const confluenceEffectiveness = [
  { label: '5M-RVOL-HIGH', winRate: 58.2, avgR: 0.56, trades: 142, contribution: 82 },
  { label: '1D-MACD_LINE-BULL', winRate: 56.8, avgR: 0.48, trades: 168, contribution: 74 },
  { label: 'GEN-TIME_OF_DAY-MORNING', winRate: 61.4, avgR: 0.62, trades: 98, contribution: 68 },
  { label: 'Both TF Conditions Met', winRate: 62.5, avgR: 0.71, trades: 112, contribution: 92 },
  { label: 'No Confluence', winRate: 42.1, avgR: -0.15, trades: 56, contribution: 18 },
];

// Equity curve with regime data
const equityCurveData = (() => {
  const points: { x: number; y: number; regime: 'bull' | 'bear' | 'neutral' }[] = [];
  let cumR = 0;
  const regimes: ('bull' | 'bear' | 'neutral')[] = ['bull', 'bull', 'bull', 'neutral', 'neutral', 'bear', 'bear', 'neutral', 'bull', 'bull', 'bull', 'bull', 'neutral', 'bull', 'bull', 'bull'];
  // Simulate 224 trades compressed into 80 points
  for (let i = 0; i < 80; i++) {
    const regime = regimes[Math.floor((i / 80) * regimes.length)];
    const base = regime === 'bull' ? 0.6 : regime === 'bear' ? -0.3 : 0.15;
    const r = base + Math.sin(i * 0.7) * 0.8 + Math.cos(i * 1.3) * 0.4;
    cumR += r;
    points.push({ x: i, y: cumR, regime });
  }
  return points;
})();

// Strategy DNA dimensions
const DNA_AXES = [
  { label: 'Win Rate', key: 'winRate' },
  { label: 'Consistency', key: 'consistency' },
  { label: 'Risk Control', key: 'risk' },
  { label: 'Speed', key: 'speed' },
  { label: 'Selectivity', key: 'selectivity' },
  { label: 'Trend Fit', key: 'trend' },
];

const dnaValues: Record<string, number> = {
  winRate: 72,
  consistency: 85,
  risk: 68,
  speed: 90,
  selectivity: 55,
  trend: 78,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const EXEC_TYPE_COLORS: Record<string, string> = {
  C: 'var(--blue)', L0: 'var(--green)', L1: '#8BC34A',
  HM: 'var(--orange)', HL: 'var(--red)',
};

function ExecBadge({ type }: { type: string }) {
  const color = EXEC_TYPE_COLORS[type] || 'var(--text-muted)';
  return (
    <span
      className="inline-flex items-center justify-center px-1.5 py-0.5 rounded text-[10px] font-mono font-bold leading-none"
      style={{
        color,
        background: `color-mix(in srgb, ${color} 15%, transparent)`,
        border: `1px solid color-mix(in srgb, ${color} 30%, transparent)`,
      }}
    >
      [{type}]
    </span>
  );
}

// ---------------------------------------------------------------------------
// SVG Sub-components
// ---------------------------------------------------------------------------

/** Health Score Ring — segmented circular gauge */
function HealthScoreRing({ score, size = 160 }: { score: number; size?: number }) {
  const c = size / 2;
  const r = (size - 20) / 2;
  const circumference = 2 * Math.PI * r;
  const arcLen = circumference * 0.75; // 270 degrees
  const fillLen = arcLen * (score / 100);
  const color = score >= 70 ? 'var(--green)' : score >= 40 ? 'var(--orange)' : 'var(--red)';
  const label = score >= 70 ? 'Healthy' : score >= 40 ? 'Moderate' : 'At Risk';

  // Segmented sections
  const segments = [
    { label: 'WR', pct: 0.25, color: 'var(--green)', value: 72 },
    { label: 'DD', pct: 0.25, color: 'var(--blue)', value: 85 },
    { label: 'Cons', pct: 0.25, color: 'var(--orange)', value: 68 },
    { label: 'Risk', pct: 0.25, color: 'var(--accent)', value: 78 },
  ];

  let segOffset = 0;

  return (
    <div className="relative flex flex-col items-center">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {/* Background track */}
        <circle
          cx={c} cy={c} r={r}
          fill="none"
          stroke="var(--bg-input)"
          strokeWidth="10"
          strokeDasharray={`${arcLen} ${circumference}`}
          strokeDashoffset={circumference * 0.125}
          strokeLinecap="round"
          transform={`rotate(135 ${c} ${c})`}
        />
        {/* Segments */}
        {segments.map((seg, i) => {
          const segLen = arcLen * seg.pct * (seg.value / 100);
          const gapLen = arcLen * seg.pct * (1 - seg.value / 100);
          const currentOffset = circumference * 0.125 - segOffset;
          segOffset += arcLen * seg.pct;
          return (
            <circle
              key={i}
              cx={c} cy={c} r={r}
              fill="none"
              stroke={seg.color}
              strokeWidth="8"
              strokeDasharray={`${segLen} ${circumference - segLen}`}
              strokeDashoffset={currentOffset}
              strokeLinecap="round"
              transform={`rotate(135 ${c} ${c})`}
              opacity={0.8}
              style={{ animation: 'health-ring-fill 1.2s ease-out' }}
            />
          );
        })}
        {/* Main fill overlay */}
        <circle
          cx={c} cy={c} r={r - 12}
          fill="none"
          stroke={color}
          strokeWidth="3"
          strokeDasharray={`${(circumference - 24 * Math.PI) * 0.75 * (score / 100)} ${circumference}`}
          strokeDashoffset={(circumference - 24 * Math.PI) * 0.125}
          strokeLinecap="round"
          transform={`rotate(135 ${c} ${c})`}
          opacity={0.5}
        />
        {/* Glow */}
        <circle
          cx={c} cy={c} r={r}
          fill="none"
          stroke={color}
          strokeWidth="14"
          strokeDasharray={`${fillLen} ${circumference}`}
          strokeDashoffset={circumference * 0.125}
          strokeLinecap="round"
          transform={`rotate(135 ${c} ${c})`}
          opacity={0.1}
          style={{ filter: 'blur(6px)' }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-3xl font-bold" style={{ color }}>{score}</span>
        <span className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>{label}</span>
      </div>
    </div>
  );
}

/** Strategy DNA — radar chart with 6 axes */
function StrategyDNAChart({ values, size = 200 }: { values: Record<string, number>; size?: number }) {
  const center = size / 2;
  const maxR = size / 2 - 24;
  const numAxes = DNA_AXES.length;
  const angleStep = (2 * Math.PI) / numAxes;
  const rings = [0.25, 0.5, 0.75, 1.0];

  const dataPoints = DNA_AXES.map((axis, i) => {
    const angle = i * angleStep - Math.PI / 2;
    const val = (values[axis.key] || 0) / 100;
    return {
      x: center + Math.cos(angle) * maxR * val,
      y: center + Math.sin(angle) * maxR * val,
      labelX: center + Math.cos(angle) * (maxR + 18),
      labelY: center + Math.sin(angle) * (maxR + 18),
      label: axis.label,
    };
  });

  const polygonPoints = dataPoints.map(p => `${p.x},${p.y}`).join(' ');

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {rings.map((ring) => {
        const pts = DNA_AXES.map((_, i) => {
          const angle = i * angleStep - Math.PI / 2;
          return `${center + Math.cos(angle) * maxR * ring},${center + Math.sin(angle) * maxR * ring}`;
        }).join(' ');
        return <polygon key={ring} points={pts} fill="none" stroke="var(--border)" strokeWidth="0.5" opacity="0.5" />;
      })}
      {DNA_AXES.map((_, i) => {
        const angle = i * angleStep - Math.PI / 2;
        return (
          <line key={i} x1={center} y1={center}
            x2={center + Math.cos(angle) * maxR} y2={center + Math.sin(angle) * maxR}
            stroke="var(--border)" strokeWidth="0.5" opacity="0.4" />
        );
      })}
      <polygon points={polygonPoints} fill="var(--accent)" fillOpacity="0.15"
        stroke="var(--accent)" strokeWidth="2"
        style={{ transformOrigin: `${center}px ${center}px`, animation: 'radar-grow 0.6s ease-out' }} />
      {dataPoints.map((p, i) => (
        <g key={i}>
          <circle cx={p.x} cy={p.y} r="4" fill="var(--accent)" />
          <circle cx={p.x} cy={p.y} r="7" fill="var(--accent)" opacity="0.2" />
        </g>
      ))}
      {dataPoints.map((p, i) => (
        <text key={`l-${i}`} x={p.labelX} y={p.labelY} textAnchor="middle" dominantBaseline="middle"
          fontSize="9" fill="var(--text-muted)" fontWeight="500">{p.label}</text>
      ))}
    </svg>
  );
}

/** Interactive Trade Timeline — horizontal scrollable */
function TradeTimeline({ trades }: { trades: typeof mockTrades }) {
  const [hoveredId, setHoveredId] = useState<number | null>(null);
  const w = Math.max(800, trades.length * 52);
  const h = 100;
  const pad = 20;

  // Compute cumulative R for y-axis
  let cumR = 0;
  const plotTrades = trades.map(t => {
    cumR += t.pnlR;
    return { ...t, cumR };
  });
  const maxCR = Math.max(...plotTrades.map(t => t.cumR), 1);
  const minCR = Math.min(...plotTrades.map(t => t.cumR), 0);
  const range = maxCR - minCR || 1;

  const points = plotTrades.map((t, i) => ({
    x: pad + (i / (plotTrades.length - 1)) * (w - pad * 2),
    y: pad + (h - pad * 2) - ((t.cumR - minCR) / range) * (h - pad * 2),
    trade: t,
  }));

  const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');

  const hoveredPoint = hoveredId !== null ? points.find(p => p.trade.id === hoveredId) : null;

  return (
    <div className="w-full overflow-x-auto">
      <svg width={w} height={h + 40} viewBox={`0 0 ${w} ${h + 40}`} style={{ minWidth: w }}>
        {/* Zero line */}
        {minCR < 0 && (
          <line x1={pad} y1={pad + (h - pad * 2) - ((0 - minCR) / range) * (h - pad * 2)}
            x2={w - pad} y2={pad + (h - pad * 2) - ((0 - minCR) / range) * (h - pad * 2)}
            stroke="var(--text-muted)" strokeWidth="0.5" strokeDasharray="4 4" opacity="0.4" />
        )}
        {/* Equity line */}
        <path d={pathD} fill="none" stroke="var(--accent)" strokeWidth="2"
          strokeLinecap="round" strokeLinejoin="round"
          style={{ strokeDasharray: 2000, animation: 'timeline-draw 1.5s ease-out forwards' }} />
        {/* Fill under curve */}
        <path d={`${pathD} L${(w - pad).toFixed(1)},${h} L${pad},${h} Z`}
          fill="var(--accent)" opacity="0.06" />
        {/* Trade dots */}
        {points.map((p, i) => {
          const isWin = p.trade.pnlR >= 0;
          const color = isWin ? 'var(--green)' : 'var(--red)';
          const isHovered = hoveredId === p.trade.id;
          return (
            <g key={i}
              onMouseEnter={() => setHoveredId(p.trade.id)}
              onMouseLeave={() => setHoveredId(null)}
              style={{ cursor: 'pointer' }}>
              <circle cx={p.x} cy={p.y} r={isHovered ? 7 : 4} fill={color} opacity={isHovered ? 1 : 0.8}
                style={{ transition: 'r 0.15s ease' }} />
              {isHovered && (
                <circle cx={p.x} cy={p.y} r="10" fill={color} opacity="0.2" />
              )}
              {/* Date labels on x-axis */}
              {i % 4 === 0 && (
                <text x={p.x} y={h + 28} textAnchor="middle" fontSize="9" fill="var(--text-muted)">
                  {p.trade.date}
                </text>
              )}
            </g>
          );
        })}
        {/* Hover tooltip */}
        {hoveredPoint && (
          <g>
            <rect
              x={hoveredPoint.x - 65} y={Math.max(0, hoveredPoint.y - 56)}
              width="130" height="48" rx="6"
              fill="var(--bg-card)" stroke="var(--border)" strokeWidth="1" />
            <text x={hoveredPoint.x} y={Math.max(14, hoveredPoint.y - 40)} textAnchor="middle"
              fontSize="10" fill="var(--text-secondary)" fontWeight="600">
              {hoveredPoint.trade.date} {hoveredPoint.trade.entryTime}
            </text>
            <text x={hoveredPoint.x} y={Math.max(28, hoveredPoint.y - 26)} textAnchor="middle"
              fontSize="10" fill={hoveredPoint.trade.pnlR >= 0 ? 'var(--green)' : 'var(--red)'} fontWeight="700">
              {hoveredPoint.trade.pnlR >= 0 ? '+' : ''}{hoveredPoint.trade.pnlR.toFixed(2)}R
            </text>
            <text x={hoveredPoint.x} y={Math.max(42, hoveredPoint.y - 12)} textAnchor="middle"
              fontSize="9" fill="var(--text-muted)">
              [{hoveredPoint.trade.execType}] {hoveredPoint.trade.exitReason}
            </text>
          </g>
        )}
      </svg>
    </div>
  );
}

/** Performance Heatmap — DoW x ToD matrix */
function PerformanceHeatmap({ data }: { data: typeof heatmapData }) {
  const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'];
  const hours = ['09:30', '10:00', '11:00', '13:00', '14:00', '15:00'];
  const cellW = 64;
  const cellH = 36;
  const labelW = 40;
  const labelH = 24;

  const maxAbsR = Math.max(...data.map(d => Math.abs(d.avgR)), 0.01);

  const getColor = (avgR: number) => {
    const intensity = Math.min(Math.abs(avgR) / maxAbsR, 1);
    if (avgR > 0) return `rgba(0, 200, 100, ${0.15 + intensity * 0.5})`;
    if (avgR < 0) return `rgba(255, 60, 60, ${0.15 + intensity * 0.5})`;
    return 'var(--bg-input)';
  };

  return (
    <div className="overflow-x-auto">
      <div style={{ display: 'inline-grid', gridTemplateColumns: `${labelW}px repeat(${hours.length}, ${cellW}px)`, gap: '2px' }}>
        {/* Header row */}
        <div />
        {hours.map(h => (
          <div key={h} className="text-center text-[10px] font-medium" style={{ color: 'var(--text-muted)', height: labelH, lineHeight: `${labelH}px` }}>
            {h}
          </div>
        ))}
        {/* Data rows */}
        {days.map(day => (
          <>
            <div key={`label-${day}`} className="text-xs font-medium flex items-center" style={{ color: 'var(--text-muted)', height: cellH }}>
              {day}
            </div>
            {hours.map(hour => {
              const cell = data.find(d => d.day === day && d.hour === hour);
              const avgR = cell?.avgR ?? 0;
              const trades = cell?.trades ?? 0;
              return (
                <div
                  key={`${day}-${hour}`}
                  className="rounded flex flex-col items-center justify-center"
                  style={{
                    background: getColor(avgR),
                    height: cellH,
                    border: '1px solid var(--border)',
                  }}
                  title={`${day} ${hour}: ${avgR >= 0 ? '+' : ''}${avgR.toFixed(2)}R (${trades} trades)`}
                >
                  <span className="text-[10px] font-mono font-bold" style={{ color: avgR >= 0 ? 'var(--green)' : 'var(--red)' }}>
                    {avgR >= 0 ? '+' : ''}{avgR.toFixed(2)}
                  </span>
                  <span className="text-[8px]" style={{ color: 'var(--text-muted)' }}>{trades}t</span>
                </div>
              );
            })}
          </>
        ))}
      </div>
    </div>
  );
}

/** Equity Curve with Regime Overlay */
function RegimeEquityCurve({ data }: { data: typeof equityCurveData }) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const w = 700;
  const h = 200;
  const pad = 30;
  const chartW = w - pad * 2;
  const chartH = h - pad * 2;

  const maxY = Math.max(...data.map(d => d.y));
  const minY = Math.min(...data.map(d => d.y));
  const rangeY = maxY - minY || 1;

  const points = data.map(d => ({
    x: pad + (d.x / (data.length - 1)) * chartW,
    y: pad + chartH - ((d.y - minY) / rangeY) * chartH,
    regime: d.regime,
    cumR: d.y,
  }));

  // Build regime bands
  const regimeBands: { x1: number; x2: number; regime: string }[] = [];
  let bandStart = 0;
  let currentRegime = data[0].regime;
  for (let i = 1; i < data.length; i++) {
    if (data[i].regime !== currentRegime || i === data.length - 1) {
      regimeBands.push({
        x1: pad + (bandStart / (data.length - 1)) * chartW,
        x2: pad + (i / (data.length - 1)) * chartW,
        regime: currentRegime,
      });
      bandStart = i;
      currentRegime = data[i].regime;
    }
  }

  const regimeColors: Record<string, string> = {
    bull: 'rgba(0, 200, 100, 0.06)',
    bear: 'rgba(255, 60, 60, 0.06)',
    neutral: 'rgba(150, 150, 150, 0.04)',
  };

  const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
  const hoveredPt = hoverIdx !== null ? points[hoverIdx] : null;

  return (
    <svg width="100%" height={h + 20} viewBox={`0 0 ${w} ${h + 20}`} preserveAspectRatio="xMidYMid meet">
      {/* Regime background bands */}
      {regimeBands.map((band, i) => (
        <rect key={i} x={band.x1} y={pad} width={band.x2 - band.x1} height={chartH}
          fill={regimeColors[band.regime]} style={{ animation: 'regime-fade 0.8s ease-out' }} />
      ))}
      {/* Grid lines */}
      {[0.25, 0.5, 0.75].map(frac => (
        <line key={frac} x1={pad} y1={pad + chartH * frac} x2={w - pad} y2={pad + chartH * frac}
          stroke="var(--border)" strokeWidth="0.5" opacity="0.3" />
      ))}
      {/* Y-axis labels */}
      <text x={pad - 4} y={pad + 4} textAnchor="end" fontSize="9" fill="var(--text-muted)">
        {maxY.toFixed(0)}R
      </text>
      <text x={pad - 4} y={pad + chartH} textAnchor="end" fontSize="9" fill="var(--text-muted)">
        {minY.toFixed(0)}R
      </text>
      {/* Equity line */}
      <path d={pathD} fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" />
      {/* Fill */}
      <path d={`${pathD} L${(w - pad).toFixed(1)},${pad + chartH} L${pad},${pad + chartH} Z`}
        fill="var(--accent)" opacity="0.08" />
      {/* Hover detection (invisible fat line) */}
      {points.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r="6" fill="transparent"
          onMouseEnter={() => setHoverIdx(i)} onMouseLeave={() => setHoverIdx(null)}
          style={{ cursor: 'crosshair' }} />
      ))}
      {/* Hover crosshair */}
      {hoveredPt && (
        <g>
          <line x1={hoveredPt.x} y1={pad} x2={hoveredPt.x} y2={pad + chartH}
            stroke="var(--text-muted)" strokeWidth="0.5" strokeDasharray="3 3" />
          <circle cx={hoveredPt.x} cy={hoveredPt.y} r="5" fill="var(--accent)" stroke="var(--bg-surface)" strokeWidth="2" />
          <rect x={hoveredPt.x - 32} y={hoveredPt.y - 22} width="64" height="18" rx="4"
            fill="var(--bg-card)" stroke="var(--border)" strokeWidth="1" />
          <text x={hoveredPt.x} y={hoveredPt.y - 10} textAnchor="middle" fontSize="10"
            fill="var(--accent)" fontWeight="700">
            {hoveredPt.cumR.toFixed(1)}R
          </text>
        </g>
      )}
      {/* Regime legend */}
      <g transform={`translate(${w - 180}, ${h + 6})`}>
        <rect x="0" y="0" width="10" height="10" rx="2" fill="rgba(0,200,100,0.3)" />
        <text x="14" y="9" fontSize="9" fill="var(--text-muted)">Bull</text>
        <rect x="50" y="0" width="10" height="10" rx="2" fill="rgba(255,60,60,0.3)" />
        <text x="64" y="9" fontSize="9" fill="var(--text-muted)">Bear</text>
        <rect x="100" y="0" width="10" height="10" rx="2" fill="rgba(150,150,150,0.2)" />
        <text x="114" y="9" fontSize="9" fill="var(--text-muted)">Neutral</text>
      </g>
    </svg>
  );
}

/** Risk/Reward Scatter Plot */
function RiskRewardScatter({ trades }: { trades: typeof mockTrades }) {
  const w = 320;
  const h = 220;
  const pad = 35;

  const maxRisk = Math.max(...trades.map(t => t.risk), 1);
  const maxReward = Math.max(...trades.map(t => Math.abs(t.reward)), 1);

  return (
    <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="xMidYMid meet">
      {/* Axes */}
      <line x1={pad} y1={h - pad} x2={w - 10} y2={h - pad} stroke="var(--border)" strokeWidth="1" />
      <line x1={pad} y1={pad - 10} x2={pad} y2={h - pad} stroke="var(--border)" strokeWidth="1" />
      {/* Zero reward line */}
      <line x1={pad} y1={pad + (h - pad * 2) / 2} x2={w - 10} y2={pad + (h - pad * 2) / 2}
        stroke="var(--text-muted)" strokeWidth="0.5" strokeDasharray="3 3" opacity="0.4" />
      {/* Axis labels */}
      <text x={w / 2} y={h - 4} textAnchor="middle" fontSize="9" fill="var(--text-muted)">Risk (R)</text>
      <text x={8} y={h / 2} textAnchor="middle" fontSize="9" fill="var(--text-muted)"
        transform={`rotate(-90 8 ${h / 2})`}>Reward (R)</text>
      {/* 1:1 line */}
      <line x1={pad} y1={pad + (h - pad * 2) / 2}
        x2={pad + (w - pad - 10) * 0.5} y2={pad - 10}
        stroke="var(--text-muted)" strokeWidth="0.5" strokeDasharray="2 2" opacity="0.3" />
      {/* 2:1 label */}
      <text x={pad + 30} y={pad + 10} fontSize="8" fill="var(--text-muted)" opacity="0.5">2:1 R:R</text>
      {/* Trade dots */}
      {trades.map((t, i) => {
        const x = pad + (t.risk / maxRisk) * (w - pad - 10);
        const midY = pad + (h - pad * 2) / 2;
        const y = midY - (t.reward / maxReward) * (h - pad * 2) / 2;
        const isWin = t.pnlR >= 0;
        return (
          <g key={i}>
            <circle cx={x} cy={y} r="5" fill={isWin ? 'var(--green)' : 'var(--red)'}
              opacity="0.7" style={{ animation: `scatter-pop 0.4s ease-out ${i * 0.05}s both` }} />
            <circle cx={x} cy={y} r="8" fill={isWin ? 'var(--green)' : 'var(--red)'} opacity="0.1" />
          </g>
        );
      })}
    </svg>
  );
}

/** Confluence Effectiveness Bars */
function ConfluenceEffectivenessChart({ data }: { data: typeof confluenceEffectiveness }) {
  const maxContrib = Math.max(...data.map(d => d.contribution));

  return (
    <div className="space-y-3">
      {data.map((item, i) => (
        <div key={i}>
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-medium truncate" style={{ color: 'var(--text-secondary)', maxWidth: '60%' }}>
              {item.label}
            </span>
            <div className="flex items-center gap-3 text-[10px]" style={{ color: 'var(--text-muted)' }}>
              <span>WR {item.winRate.toFixed(1)}%</span>
              <span style={{ color: item.avgR >= 0 ? 'var(--green)' : 'var(--red)' }}>
                {item.avgR >= 0 ? '+' : ''}{item.avgR.toFixed(2)}R
              </span>
              <span>{item.trades}t</span>
            </div>
          </div>
          <div className="h-2 rounded-full overflow-hidden" style={{ background: 'var(--bg-input)' }}>
            <div
              className="h-full rounded-full transition-all"
              style={{
                width: `${(item.contribution / maxContrib) * 100}%`,
                background: item.avgR >= 0
                  ? `linear-gradient(90deg, var(--green), var(--accent))`
                  : `linear-gradient(90deg, var(--red), var(--orange))`,
                animation: `fade-in-up 0.5s ease-out ${i * 0.1}s both`,
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

/** Live Position Widget */
function LivePositionWidget() {
  const hasPosition = true; // mock
  return (
    <div
      className="rounded-xl p-4"
      style={{
        background: 'var(--bg-card)',
        border: '1px solid var(--border)',
        animation: hasPosition ? 'glow-pulse 3s ease-in-out infinite' : undefined,
      }}
    >
      <div className="flex items-center gap-2 mb-3">
        <span
          className="w-2.5 h-2.5 rounded-full"
          style={{
            background: 'var(--green)',
            animation: 'pulse-live 2s ease-in-out infinite',
          }}
        />
        <span className="text-sm font-semibold" style={{ color: 'var(--green)' }}>
          IN POSITION
        </span>
        <ExecBadge type="C" />
      </div>
      <div className="grid grid-cols-2 gap-3 text-xs">
        <div>
          <span style={{ color: 'var(--text-muted)' }}>Entry</span>
          <p className="font-mono font-medium">$143.52</p>
        </div>
        <div>
          <span style={{ color: 'var(--text-muted)' }}>Current</span>
          <p className="font-mono font-medium" style={{ color: 'var(--green)' }}>$144.18</p>
        </div>
        <div>
          <span style={{ color: 'var(--text-muted)' }}>Stop</span>
          <p className="font-mono font-medium" style={{ color: 'var(--red)' }}>$142.85</p>
        </div>
        <div>
          <span style={{ color: 'var(--text-muted)' }}>Unrealized</span>
          <p className="font-mono font-bold" style={{ color: 'var(--green)' }}>+0.99R</p>
        </div>
        <div>
          <span style={{ color: 'var(--text-muted)' }}>Duration</span>
          <p className="font-mono">2m 14s</p>
        </div>
        <div>
          <span style={{ color: 'var(--text-muted)' }}>Target</span>
          <p className="font-mono" style={{ color: 'var(--accent)' }}>$144.86 (2R)</p>
        </div>
      </div>
      {/* Mini risk bar */}
      <div className="mt-3">
        <div className="flex justify-between text-[9px] mb-0.5" style={{ color: 'var(--text-muted)' }}>
          <span>Stop</span>
          <span>Entry</span>
          <span>Target</span>
        </div>
        <div className="h-1.5 rounded-full relative" style={{ background: 'var(--bg-input)' }}>
          {/* Stop zone */}
          <div className="absolute h-full rounded-l-full" style={{ left: 0, width: '30%', background: 'var(--red)', opacity: 0.3 }} />
          {/* Current position marker */}
          <div className="absolute h-3 w-0.5 rounded" style={{ left: '65%', top: '-3px', background: 'var(--green)' }} />
          {/* Target zone */}
          <div className="absolute h-full rounded-r-full" style={{ right: 0, width: '20%', background: 'var(--green)', opacity: 0.2 }} />
        </div>
      </div>
    </div>
  );
}

/** Smart Summary — AI-style insight */
function SmartSummary() {
  return (
    <div
      className="rounded-xl p-4"
      style={{
        background: 'linear-gradient(135deg, color-mix(in srgb, var(--accent) 8%, var(--bg-card)), var(--bg-card))',
        border: '1px solid color-mix(in srgb, var(--accent) 20%, var(--border))',
      }}
    >
      <div className="flex items-center gap-2 mb-2">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2">
          <path d="M12 2L2 7l10 5 10-5-10-5z" />
          <path d="M2 17l10 5 10-5" />
          <path d="M2 12l10 5 10-5" />
        </svg>
        <span className="text-xs font-semibold" style={{ color: 'var(--accent)' }}>Strategy Insight</span>
      </div>
      <p className="text-sm leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
        This strategy has been <strong style={{ color: 'var(--green)' }}>consistent</strong> over 224 trades
        with a 54% win rate and 2.05x profit factor. Its best performance is during
        <strong style={{ color: 'var(--accent)' }}> Tuesday mornings (09:30-10:00)</strong> with a +0.65R avg.
        The strategy struggles on <strong style={{ color: 'var(--red)' }}>Thursday afternoons</strong> where avg R
        drops to -0.35R. Consider adding a day-of-week filter to avoid low-probability sessions.
        Forward test is <strong style={{ color: 'var(--orange)' }}>tracking close to backtest</strong> (52.1%
        vs 54.0% WR) after 18 trades.
      </p>
      <span
        className="inline-block w-1.5 h-3 ml-0.5"
        style={{
          background: 'var(--accent)',
          animation: 'typing-cursor 1s step-end infinite',
          verticalAlign: 'text-bottom',
        }}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

const TABS = ['Health', 'Analytics', 'Monitoring'] as const;
type Tab = typeof TABS[number];

export default function StrategyDetailV4() {
  const [activeTab, setActiveTab] = useState<Tab>('Health');

  const healthScore = useMemo(() => {
    // Composite score from WR, PF, DD, R-squared
    const wrScore = Math.min(kpis.winRate / 60, 1) * 25;
    const pfScore = Math.min(kpis.pf / 3, 1) * 25;
    const ddScore = Math.min(Math.abs(kpis.maxDD) < 5 ? 1 : 5 / Math.abs(kpis.maxDD), 1) * 25;
    const rSqScore = kpis.rSquared * 25;
    return Math.round(wrScore + pfScore + ddScore + rSqScore);
  }, []);

  const btnStyle: React.CSSProperties = {
    background: 'var(--bg-card)',
    border: '1px solid var(--border)',
    color: 'var(--text-secondary)',
    padding: '6px 12px',
    borderRadius: '8px',
    fontSize: '0.875rem',
    cursor: 'pointer',
  };

  return (
    <div>
      <style dangerouslySetInnerHTML={{ __html: ANIMATION_STYLES }} />

      {/* Header */}
      <PageHeader
        title={strategy.name}
        subtitle={`${strategy.symbol} | ${strategy.direction} | ${strategy.timeframe} | ${strategy.session}`}
        backHref="/strategies"
        actions={
          <>
            <button style={btnStyle}>Edit</button>
            <button style={btnStyle}>Clone</button>
            <button style={{ ...btnStyle, background: 'var(--red-muted)', color: 'var(--red)', border: 'none' }}>Delete</button>
          </>
        }
      />

      {/* Inline KPI strip */}
      <div className="flex items-center gap-4 mb-4 flex-wrap">
        <span className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
          WR {kpis.winRate}% | PF {kpis.pf.toFixed(2)} | Daily R +{kpis.dailyR.toFixed(2)} | {kpis.trades} trades
        </span>
        {strategy.forwardTesting && (
          <span className="text-xs px-2 py-0.5 rounded-full font-medium"
            style={{ color: 'var(--green)', background: 'var(--green)20' }}>
            Forward Testing
          </span>
        )}
        {strategy.monitored && (
          <span className="text-xs px-2 py-0.5 rounded-full"
            style={{ color: 'var(--orange)', background: 'var(--orange)20' }}>
            Monitored
          </span>
        )}
        {strategy.confluence.map(c => (
          <span key={c} className="text-[10px] px-2 py-0.5 rounded-full"
            style={{ color: 'var(--accent)', background: 'color-mix(in srgb, var(--accent) 12%, transparent)',
              border: '1px solid color-mix(in srgb, var(--accent) 25%, transparent)' }}>
            {c}
          </span>
        ))}
      </div>

      {/* Custom tab bar */}
      <div className="flex gap-1 mb-6 p-1 rounded-lg" style={{ background: 'var(--bg-input)' }}>
        {TABS.map(tab => (
          <button
            key={tab}
            className="px-4 py-2 rounded-md text-sm font-medium transition-all"
            style={{
              background: activeTab === tab ? 'var(--bg-card)' : 'transparent',
              color: activeTab === tab ? 'var(--text-primary)' : 'var(--text-muted)',
              border: 'none',
              cursor: 'pointer',
              boxShadow: activeTab === tab ? '0 1px 3px rgba(0,0,0,0.2)' : 'none',
            }}
            onClick={() => setActiveTab(tab)}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* ================================================================== */}
      {/* HEALTH TAB                                                         */}
      {/* ================================================================== */}
      {activeTab === 'Health' && (
        <div style={{ animation: 'fade-in-up 0.3s ease-out' }}>
          {/* Top row: Health Score + DNA + Smart Summary */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-4">
            {/* Health Score */}
            <Card>
              <h3 className="text-sm font-semibold mb-2">Strategy Health</h3>
              <div className="flex justify-center">
                <HealthScoreRing score={healthScore} size={160} />
              </div>
              <div className="grid grid-cols-4 gap-2 mt-3">
                {[
                  { label: 'Win Rate', val: 72, color: 'var(--green)' },
                  { label: 'Drawdown', val: 85, color: 'var(--blue)' },
                  { label: 'Consistency', val: 68, color: 'var(--orange)' },
                  { label: 'Risk Ctrl', val: 78, color: 'var(--accent)' },
                ].map((seg, i) => (
                  <div key={i} className="text-center">
                    <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{seg.label}</div>
                    <div className="text-xs font-bold" style={{ color: seg.color }}>{seg.val}%</div>
                  </div>
                ))}
              </div>
            </Card>

            {/* Strategy DNA */}
            <Card>
              <h3 className="text-sm font-semibold mb-2">Strategy DNA</h3>
              <div className="flex justify-center">
                <StrategyDNAChart values={dnaValues} size={200} />
              </div>
            </Card>

            {/* Smart Summary */}
            <div>
              <SmartSummary />
              {/* Live Position below */}
              <div className="mt-4">
                <LivePositionWidget />
              </div>
            </div>
          </div>

          {/* Primary KPIs */}
          <div className="grid grid-cols-3 sm:grid-cols-5 lg:grid-cols-10 gap-2 mb-4">
            <MetricCard label="Win Rate" value={`${kpis.winRate.toFixed(1)}%`} positive={kpis.winRate > 50} />
            <MetricCard label="Profit Factor" value={kpis.pf.toFixed(2)} positive={kpis.pf > 1} />
            <MetricCard label="Daily R" value={`+${kpis.dailyR.toFixed(2)}`} positive />
            <MetricCard label="Trades" value={String(kpis.trades)} />
            <MetricCard label="Avg R" value={`+${kpis.avgR.toFixed(2)}`} positive />
            <MetricCard label="Total R" value={`+${kpis.totalR.toFixed(1)}`} positive />
            <MetricCard label="R-Squared" value={kpis.rSquared.toFixed(2)} positive={kpis.rSquared > 0.7} />
            <MetricCard label="Max DD" value={`${kpis.maxDD.toFixed(1)}R`} positive={false} />
            <MetricCard label="Avg Win" value={`+${kpis.avgWinR.toFixed(2)}R`} positive />
            <MetricCard label="Avg Loss" value={`${kpis.avgLossR.toFixed(2)}R`} />
          </div>

          {/* Equity Curve with Regime */}
          <Card className="mb-4">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-semibold">Equity Curve with Market Regime</h3>
              <div className="flex gap-3 text-xs" style={{ color: 'var(--text-muted)' }}>
                <span><span className="inline-block w-3 h-0.5 mr-1 rounded" style={{ background: 'var(--accent)' }} />Equity</span>
                <span><span className="inline-block w-3 h-3 mr-1 rounded" style={{ background: 'rgba(0,200,100,0.15)' }} />Bull</span>
                <span><span className="inline-block w-3 h-3 mr-1 rounded" style={{ background: 'rgba(255,60,60,0.15)' }} />Bear</span>
              </div>
            </div>
            <RegimeEquityCurve data={equityCurveData} />
          </Card>

          {/* Forward Test Comparison */}
          {strategy.forwardTesting && (
            <Card className="mb-4">
              <h3 className="text-sm font-semibold mb-3">Backtest vs Forward Test</h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
                {[
                  { m: 'Win Rate', bt: `${kpis.winRate}%`, fwd: `${fwdKpis.winRate}%`, delta: fwdKpis.winRate - kpis.winRate },
                  { m: 'PF', bt: kpis.pf.toFixed(2), fwd: fwdKpis.pf.toFixed(2), delta: fwdKpis.pf - kpis.pf },
                  { m: 'Avg R', bt: `+${kpis.avgR.toFixed(2)}`, fwd: `+${fwdKpis.avgR.toFixed(2)}`, delta: fwdKpis.avgR - kpis.avgR },
                  { m: 'Daily R', bt: `+${kpis.dailyR.toFixed(2)}`, fwd: `+${fwdKpis.dailyR.toFixed(2)}`, delta: fwdKpis.dailyR - kpis.dailyR },
                  { m: 'Total R', bt: `+${kpis.totalR.toFixed(1)}`, fwd: `+${fwdKpis.totalR.toFixed(1)}`, delta: fwdKpis.totalR - kpis.totalR },
                  { m: 'Trades', bt: String(kpis.trades), fwd: String(fwdKpis.trades), delta: 0 },
                ].map((row) => (
                  <div key={row.m} className="rounded-lg p-2 text-center"
                    style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                    <div className="text-[10px] font-medium mb-1" style={{ color: 'var(--text-muted)' }}>{row.m}</div>
                    <div className="text-xs" style={{ color: 'var(--blue)' }}>{row.bt}</div>
                    <div className="text-xs" style={{ color: 'var(--orange)' }}>{row.fwd}</div>
                    {row.delta !== 0 && (
                      <div className="text-[9px] font-mono" style={{ color: row.delta >= 0 ? 'var(--green)' : 'var(--red)' }}>
                        {row.delta >= 0 ? '+' : ''}{row.delta.toFixed(2)}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* Extended KPIs */}
          <Card>
            <h3 className="text-sm font-semibold mb-3">Extended KPIs</h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-2">
              <MetricCard label="Wins" value={String(extKpis.wins)} />
              <MetricCard label="Losses" value={String(extKpis.losses)} />
              <MetricCard label="Best Trade" value={`+${extKpis.bestTrade.toFixed(2)}R`} positive />
              <MetricCard label="Worst Trade" value={`${extKpis.worstTrade.toFixed(2)}R`} />
              <MetricCard label="Sharpe" value={extKpis.sharpe.toFixed(2)} positive={extKpis.sharpe > 1} />
              <MetricCard label="Sortino" value={extKpis.sortino.toFixed(2)} positive={extKpis.sortino > 1} />
              <MetricCard label="Calmar" value={extKpis.calmar.toFixed(2)} positive />
              <MetricCard label="Kelly" value={`${(extKpis.kelly * 100).toFixed(1)}%`} />
              <MetricCard label="Recovery" value={extKpis.recoveryFactor.toFixed(1)} positive />
              <MetricCard label="Max W Streak" value={String(extKpis.maxConsecWins)} />
              <MetricCard label="Max L Streak" value={String(extKpis.maxConsecLosses)} />
              <MetricCard label="Avg Duration" value={extKpis.avgTradeDuration} />
            </div>
          </Card>
        </div>
      )}

      {/* ================================================================== */}
      {/* ANALYTICS TAB                                                      */}
      {/* ================================================================== */}
      {activeTab === 'Analytics' && (
        <div style={{ animation: 'fade-in-up 0.3s ease-out' }}>
          {/* Trade Timeline */}
          <Card className="mb-4">
            <h3 className="text-sm font-semibold mb-2">Trade Timeline</h3>
            <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
              Hover over trades to see details. Green = win, Red = loss. Line shows cumulative R.
            </p>
            <TradeTimeline trades={mockTrades} />
          </Card>

          {/* Heatmap + Scatter side by side */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
            {/* Performance Heatmap */}
            <Card>
              <h3 className="text-sm font-semibold mb-2">Performance Heatmap</h3>
              <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
                Day-of-week vs time-of-day. Green = profitable, Red = losing.
              </p>
              <PerformanceHeatmap data={heatmapData} />
            </Card>

            {/* Risk/Reward Scatter */}
            <Card>
              <h3 className="text-sm font-semibold mb-2">Risk/Reward Scatter</h3>
              <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
                Each trade plotted by risk vs reward. Above zero = winning trade.
              </p>
              <RiskRewardScatter trades={mockTrades} />
            </Card>
          </div>

          {/* Confluence Effectiveness */}
          <Card className="mb-4">
            <h3 className="text-sm font-semibold mb-3">Confluence Effectiveness</h3>
            <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
              Which conditions contribute most to profitable trades?
            </p>
            <ConfluenceEffectivenessChart data={confluenceEffectiveness} />
          </Card>

          {/* Trade History Table */}
          <Card>
            <h3 className="text-sm font-semibold mb-3">Recent Trades</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-xs" style={{ borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid var(--border)' }}>
                    {['#', 'Date', 'Entry', 'Exit', 'Entry $', 'Exit $', 'P&L', 'Exec', 'Reason'].map(h => (
                      <th key={h} className="text-left py-2 px-2 font-medium whitespace-nowrap"
                        style={{ color: 'var(--text-muted)' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {mockTrades.map(t => (
                    <tr key={t.id} style={{ borderBottom: '1px solid var(--border)' }}
                      className="hover:bg-[var(--bg-input)]">
                      <td className="py-1.5 px-2" style={{ color: 'var(--text-muted)' }}>{t.id}</td>
                      <td className="py-1.5 px-2 whitespace-nowrap">{t.date}</td>
                      <td className="py-1.5 px-2 font-mono whitespace-nowrap">{t.entryTime}</td>
                      <td className="py-1.5 px-2 font-mono whitespace-nowrap">{t.exitTime}</td>
                      <td className="py-1.5 px-2 font-mono">${t.entryPrice.toFixed(2)}</td>
                      <td className="py-1.5 px-2 font-mono">${t.exitPrice.toFixed(2)}</td>
                      <td className="py-1.5 px-2 font-mono font-medium"
                        style={{ color: t.pnlR >= 0 ? 'var(--green)' : 'var(--red)' }}>
                        {t.pnlR >= 0 ? '+' : ''}{t.pnlR.toFixed(2)}R
                      </td>
                      <td className="py-1.5 px-2"><ExecBadge type={t.execType} /></td>
                      <td className="py-1.5 px-2">
                        <span className="text-[10px] px-1.5 py-0.5 rounded"
                          style={{
                            color: t.exitReason === 'Stop' ? 'var(--red)' : t.exitReason === 'Bar Count' ? 'var(--orange)' : 'var(--green)',
                            background: (t.exitReason === 'Stop' ? 'var(--red)' : t.exitReason === 'Bar Count' ? 'var(--orange)' : 'var(--green)') + '20',
                          }}>
                          {t.exitReason}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ================================================================== */}
      {/* MONITORING TAB                                                     */}
      {/* ================================================================== */}
      {activeTab === 'Monitoring' && (
        <div style={{ animation: 'fade-in-up 0.3s ease-out' }}>
          {/* Live Position + Alert Config side by side */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
            {/* Live Position */}
            <div>
              <h3 className="text-sm font-semibold mb-2">Live Position</h3>
              <LivePositionWidget />
            </div>

            {/* Alert Configuration */}
            <Card>
              <h3 className="text-sm font-semibold mb-3">Alert Configuration</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>Alert Tracking</span>
                  <div className="relative inline-block w-10 h-5 rounded-full transition-colors"
                    style={{ background: strategy.alertTracking ? 'var(--accent)' : 'var(--bg-input)', cursor: 'pointer' }}>
                    <div className="absolute top-0.5 w-4 h-4 rounded-full transition-transform"
                      style={{ background: 'white', left: strategy.alertTracking ? '22px' : '2px' }} />
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>Webhook URL</span>
                  <span className="text-xs font-mono truncate max-w-[200px]"
                    style={{ color: 'var(--text-muted)' }}>https://discord.com/api/we...</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>Status</span>
                  <span className="text-xs px-2 py-0.5 rounded-full font-medium"
                    style={{ color: 'var(--green)', background: 'var(--green)20' }}>Active</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>Last Alert</span>
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>2m ago</span>
                </div>
              </div>
            </Card>
          </div>

          {/* Live Chart placeholder */}
          <Card className="mb-4">
            <div className="flex items-center gap-2 mb-2">
              <span className="w-2 h-2 rounded-full" style={{ background: 'var(--green)', animation: 'pulse-live 2s ease-in-out infinite' }} />
              <h3 className="text-sm font-semibold">Live Chart</h3>
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Auto-refreshing via Polygon WS</span>
            </div>
            <div className="rounded-lg flex items-center justify-center" style={{ background: 'var(--bg-input)', height: 350 }}>
              <div className="text-center">
                <svg width="48" height="48" viewBox="0 0 48 48" fill="none" className="mx-auto mb-2">
                  <path d="M4 36 L12 24 L20 30 L28 14 L36 22 L44 8" stroke="var(--accent)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M4 36 L12 24 L20 30 L28 14 L36 22 L44 8 L44 40 L4 40 Z" fill="var(--accent)" opacity="0.08" />
                  {/* Candles */}
                  <rect x="10" y="20" width="3" height="12" rx="0.5" fill="var(--green)" opacity="0.5" />
                  <rect x="18" y="26" width="3" height="8" rx="0.5" fill="var(--red)" opacity="0.5" />
                  <rect x="26" y="12" width="3" height="14" rx="0.5" fill="var(--green)" opacity="0.5" />
                  <rect x="34" y="18" width="3" height="10" rx="0.5" fill="var(--green)" opacity="0.5" />
                </svg>
                <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Live OHLC chart with trade markers</p>
                <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>Connected to Polygon.io WebSocket</p>
              </div>
            </div>
          </Card>

          {/* Recent Alerts */}
          <Card className="mb-4">
            <h3 className="text-sm font-semibold mb-3">Recent Alerts</h3>
            <div className="space-y-2">
              {[
                { time: '09:42:15', type: 'ENTRY', price: '$142.35', status: 'Delivered', color: 'var(--green)' },
                { time: '09:46:32', type: 'EXIT', price: '$142.98', status: 'Delivered', color: 'var(--blue)' },
                { time: '10:15:03', type: 'ENTRY', price: '$143.10', status: 'Delivered', color: 'var(--green)' },
                { time: '10:19:18', type: 'EXIT', price: '$142.65', status: 'Delivered', color: 'var(--red)' },
                { time: '11:02:44', type: 'ENTRY', price: '$143.50', status: 'Failed', color: 'var(--green)' },
              ].map((alert, i) => (
                <div key={i} className="flex items-center justify-between py-2 px-3 rounded-lg"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                  <div className="flex items-center gap-3">
                    <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>{alert.time}</span>
                    <span className="text-xs font-medium px-2 py-0.5 rounded"
                      style={{ color: alert.color, background: alert.color + '20' }}>
                      {alert.type}
                    </span>
                    <span className="text-xs font-mono">{alert.price}</span>
                  </div>
                  <span className="text-[10px] px-2 py-0.5 rounded-full"
                    style={{
                      color: alert.status === 'Delivered' ? 'var(--green)' : 'var(--red)',
                      background: (alert.status === 'Delivered' ? 'var(--green)' : 'var(--red)') + '15',
                    }}>
                    {alert.status}
                  </span>
                </div>
              ))}
            </div>
          </Card>

          {/* Alert Accuracy */}
          <Card>
            <h3 className="text-sm font-semibold mb-3">Alert vs Backtest Accuracy</h3>
            <div className="grid grid-cols-3 gap-3 mb-4">
              <MetricCard label="Match Rate" value="92.3%" positive />
              <MetricCard label="Avg Drift" value="$0.02" />
              <MetricCard label="Max Drift" value="$0.05" />
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs" style={{ borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border)' }}>
                    {['Time', 'Type', 'Alert $', 'Backtest $', 'Drift', 'Match'].map(h => (
                      <th key={h} className="text-left py-2 px-2" style={{ color: 'var(--text-muted)' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[
                    { time: '09:42', type: 'ENTRY', alert: '$142.35', bt: '$142.37', drift: '$0.02', match: true },
                    { time: '09:46', type: 'EXIT', alert: '$142.98', bt: '$142.95', drift: '$0.03', match: true },
                    { time: '10:15', type: 'ENTRY', alert: '$143.10', bt: '$143.10', drift: '$0.00', match: true },
                    { time: '10:19', type: 'EXIT', alert: '$142.65', bt: '$142.62', drift: '$0.03', match: true },
                    { time: '11:02', type: 'ENTRY', alert: '$143.50', bt: '$143.48', drift: '$0.02', match: false },
                  ].map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                      <td className="py-1.5 px-2 font-mono">{row.time}</td>
                      <td className="py-1.5 px-2">{row.type}</td>
                      <td className="py-1.5 px-2 font-mono">{row.alert}</td>
                      <td className="py-1.5 px-2 font-mono">{row.bt}</td>
                      <td className="py-1.5 px-2 font-mono">{row.drift}</td>
                      <td className="py-1.5 px-2">
                        <span style={{ color: row.match ? 'var(--green)' : 'var(--red)' }}>
                          {row.match ? 'Yes' : 'No'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* Configuration footer */}
      <Card className="mt-4">
        <h3 className="text-sm font-semibold mb-2">Strategy Configuration</h3>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
          <div>
            <span style={{ color: 'var(--text-muted)' }}>Entry Trigger</span>
            <p className="font-medium">{strategy.trigger}</p>
          </div>
          <div>
            <span style={{ color: 'var(--text-muted)' }}>Exit Trigger</span>
            <p className="font-medium">{strategy.exitTrigger}</p>
          </div>
          <div>
            <span style={{ color: 'var(--text-muted)' }}>Stop Method</span>
            <p className="font-medium">{strategy.stop}</p>
          </div>
          <div>
            <span style={{ color: 'var(--text-muted)' }}>Target</span>
            <p className="font-medium">{strategy.target}</p>
          </div>
          <div>
            <span style={{ color: 'var(--text-muted)' }}>Backtest Period</span>
            <p className="font-medium">{strategy.btStart} to {strategy.btEnd} ({strategy.btDays}d)</p>
          </div>
          <div>
            <span style={{ color: 'var(--text-muted)' }}>TF Confluence</span>
            <p className="font-medium">{strategy.confluence.join(', ')}</p>
          </div>
          <div>
            <span style={{ color: 'var(--text-muted)' }}>General Confluence</span>
            <p className="font-medium">{strategy.generalConfluence.join(', ')}</p>
          </div>
          <div>
            <span style={{ color: 'var(--text-muted)' }}>Forward Since</span>
            <p className="font-medium">{strategy.forwardTestStart}</p>
          </div>
        </div>
      </Card>
    </div>
  );
}
