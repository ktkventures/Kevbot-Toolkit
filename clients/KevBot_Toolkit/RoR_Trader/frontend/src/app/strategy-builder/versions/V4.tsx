'use client';

import { useState, useMemo, useCallback } from 'react';
import Card from '@/components/Card';
import MetricCard from '@/components/MetricCard';
import Modal from '@/components/Modal';

// =============================================================================
// V4 Meta: Creative — Visual Pipeline Builder + Strategy DNA
// =============================================================================
// The "wow factor" version. Key creative features:
//
// 1. VISUAL PIPELINE — Data -> Indicators -> Confluence -> Entry -> Position -> Exit
//    rendered as an interactive node graph with connecting lines
// 2. STRATEGY DNA — visual fingerprint radar chart showing the strategy's character
// 3. QUICK PRESETS — one-click strategy templates that fill the pipeline
// 4. AI INSIGHT — mock AI suggestion card with animated typing effect
// 5. CONFIDENCE METER — animated gauge based on trade count, win rate stability
// 6. INTERACTIVE EQUITY CURVE — SVG with hover tooltips per trade
// 7. RISK CALCULATOR — live risk/reward projection with account sizing
// 8. LIVE INDICATOR PREVIEW — mini sparklines update as config changes
// =============================================================================

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TriggerDef {
  id: string;
  name: string;
  execType: 'C' | 'L0' | 'L1' | 'HM' | 'HL';
  pack: string;
}

interface ConfluenceCondition {
  id: string;
  label: string;
  pack: string;
}

interface TradeRow {
  id: number;
  entryTime: string;
  exitTime: string;
  direction: 'LONG' | 'SHORT';
  entryPrice: number;
  exitPrice: number;
  rMultiple: number;
  execType: string;
  exitReason: string;
}

interface PipelineNode {
  id: string;
  label: string;
  sublabel: string;
  status: 'active' | 'configured' | 'empty';
  icon: string;
}

// ---------------------------------------------------------------------------
// Mock Data
// ---------------------------------------------------------------------------

const TIMEFRAMES = ['1Min', '5Min', '15Min', '1H', '4H', '1D'];
const DIRECTIONS = ['LONG', 'SHORT'];
const SESSIONS = ['RTH', 'Pre-Market', 'After Hours', 'Extended', '24/7'];
const STOP_METHODS = ['Swing', 'ATR', 'Fixed $', 'Pct %'];
const TARGET_METHODS = ['None', 'R:R', 'ATR', 'Fixed $', 'Pct %'];

const MOCK_TRIGGERS: TriggerDef[] = [
  { id: 'ema-price-cross-up', name: 'EMA Price Cross Up', execType: 'C', pack: 'EMA Stack' },
  { id: 'ema-price-cross-down', name: 'EMA Price Cross Down', execType: 'C', pack: 'EMA Stack' },
  { id: 'ema-fan-open-bull', name: 'EMA Fan Open Bull', execType: 'C', pack: 'EMA Stack' },
  { id: 'macd-cross-bull', name: 'MACD Cross Bull', execType: 'C', pack: 'MACD' },
  { id: 'macd-cross-bear', name: 'MACD Cross Bear', execType: 'C', pack: 'MACD' },
  { id: 'macd-zero-cross-up', name: 'MACD Zero Cross Up', execType: 'C', pack: 'MACD' },
  { id: 'vwap-cross-up', name: 'VWAP Cross Up', execType: 'L0', pack: 'VWAP' },
  { id: 'vwap-cross-down', name: 'VWAP Cross Down', execType: 'L0', pack: 'VWAP' },
  { id: 'rsi-oversold-exit', name: 'RSI Oversold Exit', execType: 'C', pack: 'RSI' },
  { id: 'ut-bot-buy', name: 'UT Bot Buy', execType: 'HM', pack: 'UT Bot' },
  { id: 'ut-bot-sell', name: 'UT Bot Sell', execType: 'HM', pack: 'UT Bot' },
  { id: 'supertrend-flip-bull', name: 'Supertrend Flip Bull', execType: 'L1', pack: 'Supertrend' },
  { id: 'supertrend-flip-bear', name: 'Supertrend Flip Bear', execType: 'L1', pack: 'Supertrend' },
  { id: 'keltner-bounce-long', name: 'Keltner Bounce Long', execType: 'HL', pack: 'Keltner' },
];

const ALL_CONFLUENCE: ConfluenceCondition[] = [
  { id: '5M-EMA_STACK-BULL', label: '5M EMA Stack Bullish', pack: 'EMA Stack' },
  { id: '5M-EMA_STACK-SML', label: '5M EMA Stack S>M>L', pack: 'EMA Stack' },
  { id: '15M-EMA_STACK-BULL', label: '15M EMA Stack Bullish', pack: 'EMA Stack' },
  { id: '1H-EMA_STACK-BULL', label: '1H EMA Stack Bullish', pack: 'EMA Stack' },
  { id: '1D-MACD_LINE-BULL', label: '1D MACD Line Bullish', pack: 'MACD' },
  { id: '5M-MACD_LINE-BULL', label: '5M MACD Line Bullish', pack: 'MACD' },
  { id: '1H-VWAP_POS-ABOVE', label: '1H Above VWAP', pack: 'VWAP' },
  { id: '15M-SUPERTREND-BULL', label: '15M Supertrend Bullish', pack: 'Supertrend' },
  { id: '1H-SUPERTREND-BULL', label: '1H Supertrend Bullish', pack: 'Supertrend' },
  { id: 'GEN-TOD-NY_MORNING', label: 'NY Morning (9:30-11:30)', pack: 'Time / Session' },
  { id: 'GEN-DOW-MON_TUE_WED', label: 'Mon/Tue/Wed', pack: 'Time / Session' },
  { id: 'GEN-SESSION-RTH', label: 'Regular Trading Hours', pack: 'Time / Session' },
];

const MOCK_TRADES: TradeRow[] = [
  { id: 1, entryTime: '03/18 09:35', exitTime: '03/18 10:12', direction: 'LONG', entryPrice: 567.42, exitPrice: 569.18, rMultiple: 1.84, execType: 'C', exitReason: 'signal' },
  { id: 2, entryTime: '03/18 10:45', exitTime: '03/18 11:30', direction: 'LONG', entryPrice: 568.90, exitPrice: 567.15, rMultiple: -1.00, execType: 'C', exitReason: 'stop' },
  { id: 3, entryTime: '03/18 13:15', exitTime: '03/18 14:02', direction: 'LONG', entryPrice: 567.80, exitPrice: 570.25, rMultiple: 2.56, execType: 'L0', exitReason: 'target' },
  { id: 4, entryTime: '03/19 09:32', exitTime: '03/19 09:58', direction: 'LONG', entryPrice: 571.10, exitPrice: 570.20, rMultiple: -0.85, execType: 'HM', exitReason: 'stop' },
  { id: 5, entryTime: '03/19 10:15', exitTime: '03/19 11:45', direction: 'LONG', entryPrice: 570.55, exitPrice: 573.40, rMultiple: 3.12, execType: 'C', exitReason: 'signal' },
  { id: 6, entryTime: '03/19 13:05', exitTime: '03/19 13:42', direction: 'LONG', entryPrice: 572.80, exitPrice: 573.60, rMultiple: 0.84, execType: 'C', exitReason: 'bar_count' },
  { id: 7, entryTime: '03/19 14:10', exitTime: '03/19 15:15', direction: 'LONG', entryPrice: 573.20, exitPrice: 572.05, rMultiple: -1.00, execType: 'L1', exitReason: 'stop' },
  { id: 8, entryTime: '03/20 09:35', exitTime: '03/20 10:50', direction: 'LONG', entryPrice: 570.15, exitPrice: 572.90, rMultiple: 2.89, execType: 'C', exitReason: 'target' },
  { id: 9, entryTime: '03/20 11:20', exitTime: '03/20 12:05', direction: 'LONG', entryPrice: 572.45, exitPrice: 573.80, rMultiple: 1.41, execType: 'HM', exitReason: 'signal' },
  { id: 10, entryTime: '03/20 13:30', exitTime: '03/20 14:45', direction: 'LONG', entryPrice: 573.10, exitPrice: 571.90, rMultiple: -1.00, execType: 'C', exitReason: 'stop' },
  { id: 11, entryTime: '03/20 15:00', exitTime: '03/20 15:45', direction: 'LONG', entryPrice: 572.20, exitPrice: 574.50, rMultiple: 2.41, execType: 'C', exitReason: 'signal' },
  { id: 12, entryTime: '03/21 09:33', exitTime: '03/21 10:20', direction: 'LONG', entryPrice: 574.80, exitPrice: 575.95, rMultiple: 1.20, execType: 'L0', exitReason: 'signal' },
];

const MOCK_KPIS = {
  winRate: 58.3, profitFactor: 2.14, dailyR: 0.47, totalTrades: 127,
  avgR: 0.22, totalR: 28.4, rSquared: 0.87, maxDD: -4.2,
  sharpe: 1.67, expectancy: 0.53, avgWin: 1.82, avgLoss: -0.94,
};

const STRATEGY_PRESETS = [
  { name: 'Momentum Scalper', ticker: 'SPY', direction: 'LONG', timeframe: '1Min', trigger: 'ema-price-cross-up', confluences: ['5M-EMA_STACK-BULL', 'GEN-TOD-NY_MORNING'], stop: 'ATR', target: 'R:R', description: 'Fast EMA crosses with bullish stack confirmation during NY morning' },
  { name: 'VWAP Mean Reversion', ticker: 'QQQ', direction: 'LONG', timeframe: '5Min', trigger: 'vwap-cross-up', confluences: ['1H-VWAP_POS-ABOVE', '5M-MACD_LINE-BULL'], stop: 'Swing', target: 'R:R', description: 'VWAP bounces with momentum confirmation on 5-min chart' },
  { name: 'Trend Following', ticker: 'NVDA', direction: 'LONG', timeframe: '15Min', trigger: 'supertrend-flip-bull', confluences: ['1H-EMA_STACK-BULL', '1D-MACD_LINE-BULL', '1H-SUPERTREND-BULL'], stop: 'ATR', target: 'None', description: 'Multi-timeframe trend alignment with supertrend entries' },
  { name: 'Hybrid Breakout', ticker: 'TSLA', direction: 'LONG', timeframe: '5Min', trigger: 'ut-bot-buy', confluences: ['5M-EMA_STACK-SML', 'GEN-SESSION-RTH'], stop: 'Swing', target: 'ATR', description: 'UT Bot hybrid entries with full EMA alignment during RTH' },
];

// Strategy DNA dimensions (radar chart axes)
const DNA_AXES = [
  { label: 'Aggression', key: 'aggression' },
  { label: 'Selectivity', key: 'selectivity' },
  { label: 'Speed', key: 'speed' },
  { label: 'Consistency', key: 'consistency' },
  { label: 'Risk Control', key: 'risk' },
  { label: 'Trend Alignment', key: 'trend' },
];

// ---------------------------------------------------------------------------
// CSS Keyframes (injected via style tag)
// ---------------------------------------------------------------------------

const ANIMATION_STYLES = `
@keyframes pipeline-pulse {
  0%, 100% { opacity: 0.4; }
  50% { opacity: 1; }
}
@keyframes flow-line {
  0% { stroke-dashoffset: 24; }
  100% { stroke-dashoffset: 0; }
}
@keyframes confidence-fill {
  0% { stroke-dasharray: 0 314; }
}
@keyframes fade-in-up {
  0% { opacity: 0; transform: translateY(8px); }
  100% { opacity: 1; transform: translateY(0); }
}
@keyframes glow-pulse {
  0%, 100% { box-shadow: 0 0 8px rgba(0,255,136,0.15); }
  50% { box-shadow: 0 0 20px rgba(0,255,136,0.35); }
}
@keyframes typing-cursor {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}
@keyframes sparkline-draw {
  0% { stroke-dashoffset: 200; }
  100% { stroke-dashoffset: 0; }
}
@keyframes radar-grow {
  0% { transform: scale(0); opacity: 0; }
  100% { transform: scale(1); opacity: 1; }
}
`;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const EXEC_TYPE_COLORS: Record<string, string> = {
  C: 'var(--blue)', L0: 'var(--green)', L1: 'var(--orange)',
  HM: 'var(--accent)', HL: 'var(--red)',
};

const EXIT_REASON_COLORS: Record<string, string> = {
  signal: 'var(--green)', target: 'var(--green)',
  stop: 'var(--red)', bar_count: 'var(--orange)',
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
// SVG Components
// ---------------------------------------------------------------------------

/** Visual Pipeline — horizontal node graph with animated flow lines */
function PipelineGraph({
  nodes,
  onNodeClick,
}: {
  nodes: PipelineNode[];
  onNodeClick: (id: string) => void;
}) {
  const nodeW = 120;
  const nodeH = 70;
  const gap = 40;
  const totalW = nodes.length * nodeW + (nodes.length - 1) * gap;
  const svgH = nodeH + 20;

  const statusColors: Record<string, { fill: string; stroke: string; glow: string }> = {
    active: { fill: 'var(--accent)', stroke: 'var(--accent)', glow: 'rgba(0,255,136,0.3)' },
    configured: { fill: 'var(--bg-card)', stroke: 'var(--green)', glow: 'rgba(76,175,80,0.15)' },
    empty: { fill: 'var(--bg-card)', stroke: 'var(--border)', glow: 'none' },
  };

  return (
    <div className="w-full overflow-x-auto py-2">
      <svg width={totalW} height={svgH} viewBox={`0 0 ${totalW} ${svgH}`} style={{ minWidth: totalW }}>
        {/* Flow lines between nodes */}
        {nodes.slice(0, -1).map((_, i) => {
          const x1 = i * (nodeW + gap) + nodeW;
          const x2 = (i + 1) * (nodeW + gap);
          const y = svgH / 2;
          return (
            <line
              key={`line-${i}`}
              x1={x1} y1={y} x2={x2} y2={y}
              stroke="var(--accent)"
              strokeWidth="2"
              strokeDasharray="6 6"
              opacity="0.5"
              style={{ animation: 'flow-line 1.5s linear infinite' }}
            />
          );
        })}

        {/* Arrow heads */}
        {nodes.slice(0, -1).map((_, i) => {
          const x = (i + 1) * (nodeW + gap) - 6;
          const y = svgH / 2;
          return (
            <polygon
              key={`arrow-${i}`}
              points={`${x},${y - 4} ${x + 8},${y} ${x},${y + 4}`}
              fill="var(--accent)"
              opacity="0.6"
            />
          );
        })}

        {/* Nodes */}
        {nodes.map((node, i) => {
          const x = i * (nodeW + gap);
          const y = (svgH - nodeH) / 2;
          const colors = statusColors[node.status];
          return (
            <g key={node.id} onClick={() => onNodeClick(node.id)} style={{ cursor: 'pointer' }}>
              {/* Glow */}
              {node.status !== 'empty' && (
                <rect
                  x={x - 2} y={y - 2} width={nodeW + 4} height={nodeH + 4} rx="14"
                  fill="none" stroke={colors.glow} strokeWidth="3"
                  style={{ animation: node.status === 'active' ? 'pipeline-pulse 2s ease-in-out infinite' : undefined }}
                />
              )}
              {/* Background */}
              <rect
                x={x} y={y} width={nodeW} height={nodeH} rx="12"
                fill={colors.fill}
                stroke={colors.stroke}
                strokeWidth="1.5"
              />
              {/* Icon */}
              <text
                x={x + nodeW / 2} y={y + 22}
                textAnchor="middle" fontSize="16"
                fill={node.status === 'active' ? '#000' : 'var(--text-primary)'}
              >
                {node.icon}
              </text>
              {/* Label */}
              <text
                x={x + nodeW / 2} y={y + 40}
                textAnchor="middle" fontSize="11" fontWeight="600"
                fill={node.status === 'active' ? '#000' : 'var(--text-primary)'}
              >
                {node.label}
              </text>
              {/* Sublabel */}
              <text
                x={x + nodeW / 2} y={y + 54}
                textAnchor="middle" fontSize="9"
                fill={node.status === 'active' ? 'rgba(0,0,0,0.6)' : 'var(--text-muted)'}
              >
                {node.sublabel}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/** Strategy DNA — radar chart with 6 axes */
function StrategyDNAChart({
  values,
  size = 200,
}: {
  values: Record<string, number>; // 0-100 for each axis
  size?: number;
}) {
  const center = size / 2;
  const maxR = size / 2 - 20;
  const numAxes = DNA_AXES.length;
  const angleStep = (2 * Math.PI) / numAxes;

  // Grid rings
  const rings = [0.25, 0.5, 0.75, 1.0];

  // Compute data polygon points
  const dataPoints = DNA_AXES.map((axis, i) => {
    const angle = i * angleStep - Math.PI / 2;
    const val = (values[axis.key] || 0) / 100;
    return {
      x: center + Math.cos(angle) * maxR * val,
      y: center + Math.sin(angle) * maxR * val,
      labelX: center + Math.cos(angle) * (maxR + 16),
      labelY: center + Math.sin(angle) * (maxR + 16),
      label: axis.label,
    };
  });

  const polygonPoints = dataPoints.map(p => `${p.x},${p.y}`).join(' ');

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {/* Grid rings */}
      {rings.map((ring) => {
        const pts = DNA_AXES.map((_, i) => {
          const angle = i * angleStep - Math.PI / 2;
          return `${center + Math.cos(angle) * maxR * ring},${center + Math.sin(angle) * maxR * ring}`;
        }).join(' ');
        return (
          <polygon
            key={ring}
            points={pts}
            fill="none"
            stroke="var(--border)"
            strokeWidth="0.5"
            opacity="0.5"
          />
        );
      })}

      {/* Axis lines */}
      {DNA_AXES.map((_, i) => {
        const angle = i * angleStep - Math.PI / 2;
        return (
          <line
            key={i}
            x1={center} y1={center}
            x2={center + Math.cos(angle) * maxR}
            y2={center + Math.sin(angle) * maxR}
            stroke="var(--border)"
            strokeWidth="0.5"
            opacity="0.4"
          />
        );
      })}

      {/* Data fill */}
      <polygon
        points={polygonPoints}
        fill="var(--accent)"
        fillOpacity="0.15"
        stroke="var(--accent)"
        strokeWidth="2"
        style={{ transformOrigin: `${center}px ${center}px`, animation: 'radar-grow 0.6s ease-out' }}
      />

      {/* Data points */}
      {dataPoints.map((p, i) => (
        <g key={i}>
          <circle cx={p.x} cy={p.y} r="4" fill="var(--accent)" />
          <circle cx={p.x} cy={p.y} r="7" fill="var(--accent)" opacity="0.2" />
        </g>
      ))}

      {/* Axis labels */}
      {dataPoints.map((p, i) => (
        <text
          key={`label-${i}`}
          x={p.labelX}
          y={p.labelY}
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize="9"
          fill="var(--text-muted)"
          fontWeight="500"
        >
          {p.label}
        </text>
      ))}
    </svg>
  );
}

/** Confidence Gauge — circular meter */
function ConfidenceGauge({ score, size = 100 }: { score: number; size?: number }) {
  const r = (size - 12) / 2;
  const c = size / 2;
  const circumference = 2 * Math.PI * r;
  // Only show 270 degrees (three-quarter arc)
  const arcLength = circumference * 0.75;
  const fillLength = arcLength * (score / 100);
  const color = score >= 70 ? 'var(--green)' : score >= 40 ? 'var(--orange)' : 'var(--red)';
  const label = score >= 70 ? 'High' : score >= 40 ? 'Medium' : 'Low';

  return (
    <div className="relative flex flex-col items-center">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {/* Background arc */}
        <circle
          cx={c} cy={c} r={r}
          fill="none"
          stroke="var(--bg-input)"
          strokeWidth="6"
          strokeDasharray={`${arcLength} ${circumference}`}
          strokeDashoffset={circumference * 0.125}
          strokeLinecap="round"
          transform={`rotate(135 ${c} ${c})`}
        />
        {/* Filled arc */}
        <circle
          cx={c} cy={c} r={r}
          fill="none"
          stroke={color}
          strokeWidth="6"
          strokeDasharray={`${fillLength} ${circumference}`}
          strokeDashoffset={circumference * 0.125}
          strokeLinecap="round"
          transform={`rotate(135 ${c} ${c})`}
          style={{ animation: 'confidence-fill 1s ease-out' }}
        />
        {/* Glow */}
        <circle
          cx={c} cy={c} r={r}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeDasharray={`${fillLength} ${circumference}`}
          strokeDashoffset={circumference * 0.125}
          strokeLinecap="round"
          transform={`rotate(135 ${c} ${c})`}
          opacity="0.2"
          style={{ filter: 'blur(3px)' }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-2xl font-bold" style={{ color }}>{score}</span>
        <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{label}</span>
      </div>
    </div>
  );
}

/** Interactive Equity Curve with hover tooltips */
function InteractiveEquityCurve({
  trades,
  width = 600,
  height = 160,
}: {
  trades: TradeRow[];
  width?: number;
  height?: number;
}) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const equityData = useMemo(() => {
    let cumR = 0;
    return trades.map((t) => {
      cumR += t.rMultiple;
      return { ...t, cumR };
    });
  }, [trades]);

  const maxR = Math.max(...equityData.map(d => d.cumR), 1);
  const minR = Math.min(...equityData.map(d => d.cumR), 0);
  const range = maxR - minR || 1;
  const pad = 8;
  const chartW = width - pad * 2;
  const chartH = height - pad * 2 - 20;

  const points = equityData.map((d, i) => ({
    x: pad + (i / (equityData.length - 1)) * chartW,
    y: pad + chartH - ((d.cumR - minR) / range) * chartH,
    data: d,
  }));

  const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`).join(' ');
  const fillD = `${pathD} L${pad + chartW},${pad + chartH} L${pad},${pad + chartH} Z`;
  const finalVal = equityData[equityData.length - 1]?.cumR ?? 0;
  const color = finalVal >= 0 ? 'var(--green)' : 'var(--red)';

  return (
    <div className="relative">
      <svg
        width="100%"
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMidYMid meet"
        style={{ overflow: 'visible' }}
      >
        <defs>
          <linearGradient id="eqGradV4" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.2" />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </linearGradient>
        </defs>
        {/* Zero line */}
        {minR < 0 && (
          <line
            x1={pad} y1={pad + chartH - ((0 - minR) / range) * chartH}
            x2={pad + chartW} y2={pad + chartH - ((0 - minR) / range) * chartH}
            stroke="var(--text-muted)" strokeWidth="0.5" strokeDasharray="3 3" opacity="0.4"
          />
        )}
        {/* Fill */}
        <path d={fillD} fill="url(#eqGradV4)" />
        {/* Line */}
        <path
          d={pathD} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round"
          style={{ strokeDasharray: 500, animation: 'sparkline-draw 1.2s ease-out forwards' }}
        />

        {/* Trade dots */}
        {points.map((p, i) => (
          <g
            key={i}
            onMouseEnter={() => setHoverIdx(i)}
            onMouseLeave={() => setHoverIdx(null)}
            style={{ cursor: 'pointer' }}
          >
            <circle cx={p.x} cy={p.y} r="12" fill="transparent" />
            <circle
              cx={p.x} cy={p.y}
              r={hoverIdx === i ? 5 : 3}
              fill={p.data.rMultiple >= 0 ? 'var(--green)' : 'var(--red)'}
              stroke="var(--bg-card)" strokeWidth="1.5"
              style={{ transition: 'r 0.15s ease' }}
            />
          </g>
        ))}

        {/* Hover tooltip */}
        {hoverIdx !== null && points[hoverIdx] && (() => {
          const p = points[hoverIdx];
          const tooltipW = 130;
          const tooltipH = 52;
          let tx = p.x - tooltipW / 2;
          if (tx < 0) tx = 0;
          if (tx + tooltipW > width) tx = width - tooltipW;
          const ty = p.y - tooltipH - 12;
          return (
            <g>
              <rect x={tx} y={ty} width={tooltipW} height={tooltipH} rx="6"
                fill="var(--bg-secondary)" stroke="var(--border)" strokeWidth="1" />
              <text x={tx + 8} y={ty + 16} fontSize="10" fill="var(--text-muted)">
                {p.data.entryTime}
              </text>
              <text x={tx + 8} y={ty + 30} fontSize="11" fontWeight="600"
                fill={p.data.rMultiple >= 0 ? 'var(--green)' : 'var(--red)'}>
                {p.data.rMultiple >= 0 ? '+' : ''}{p.data.rMultiple.toFixed(2)}R
              </text>
              <text x={tx + 8} y={ty + 44} fontSize="9" fill="var(--text-muted)">
                {p.data.exitReason} | Cum: {p.data.cumR.toFixed(1)}R
              </text>
            </g>
          );
        })()}
      </svg>
    </div>
  );
}

/** Mini Sparkline for indicator preview */
function IndicatorSparkline({
  data,
  color,
  label,
  width = 120,
  height = 32,
}: {
  data: number[];
  color: string;
  label: string;
  width?: number;
  height?: number;
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
    <div className="flex items-center gap-2">
      <span className="text-[10px] w-10 text-right" style={{ color: 'var(--text-muted)' }}>{label}</span>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        <path
          d={pathD} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round"
          style={{ strokeDasharray: 200, animation: 'sparkline-draw 0.8s ease-out forwards' }}
        />
      </svg>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

/** AI Insight Card with typing effect */
function AIInsightCard({ ticker, direction }: { ticker: string; direction: string }) {
  const insight = `Based on ${ticker}'s recent price action, the EMA Stack is in a strong bullish alignment on the 5M chart. Consider adding VWAP > zone confluence for higher selectivity. Win rate typically improves 4-6% with multi-timeframe confirmation.`;

  return (
    <Card>
      <div className="flex items-center gap-2 mb-3">
        <div
          className="w-7 h-7 rounded-lg flex items-center justify-center text-xs font-bold"
          style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
        >
          AI
        </div>
        <span className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>Strategy Insight</span>
        <span
          className="text-[10px] px-1.5 py-0.5 rounded-full"
          style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
        >
          Mock
        </span>
      </div>
      <p className="text-xs leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
        {insight}
        <span
          className="inline-block w-1.5 h-3 ml-0.5"
          style={{ background: 'var(--accent)', animation: 'typing-cursor 0.8s step-end infinite' }}
        />
      </p>
      <div className="flex gap-2 mt-3">
        <button
          className="text-[10px] px-2.5 py-1 rounded-md"
          style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
        >
          Apply Suggestion
        </button>
        <button
          className="text-[10px] px-2.5 py-1 rounded-md"
          style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}
        >
          Dismiss
        </button>
      </div>
    </Card>
  );
}

/** Risk Calculator */
function RiskCalculator({ kpis }: { kpis: typeof MOCK_KPIS }) {
  const [accountSize, setAccountSize] = useState(25000);
  const [riskPerTrade, setRiskPerTrade] = useState(100);

  const riskPct = ((riskPerTrade / accountSize) * 100).toFixed(2);
  const expectedDaily = (kpis.dailyR * riskPerTrade).toFixed(0);
  const expectedMonthly = (kpis.dailyR * riskPerTrade * 21).toFixed(0);
  const worstDrawdown = (kpis.maxDD * riskPerTrade).toFixed(0);
  const tradesPerDay = (kpis.totalTrades / 30).toFixed(1);

  return (
    <Card>
      <h3 className="text-xs font-semibold mb-3" style={{ color: 'var(--text-muted)' }}>Risk Calculator</h3>
      <div className="grid grid-cols-2 gap-3 mb-3">
        <div>
          <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>Account Size</label>
          <div className="flex items-center gap-1">
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>$</span>
            <input
              type="number"
              className="w-full px-2 py-1.5 rounded-lg text-sm"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
              value={accountSize}
              onChange={(e) => setAccountSize(parseInt(e.target.value) || 0)}
            />
          </div>
        </div>
        <div>
          <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>Risk per Trade (1R)</label>
          <div className="flex items-center gap-1">
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>$</span>
            <input
              type="number"
              className="w-full px-2 py-1.5 rounded-lg text-sm"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
              value={riskPerTrade}
              onChange={(e) => setRiskPerTrade(parseInt(e.target.value) || 0)}
            />
          </div>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-2">
        {[
          { label: 'Risk %', value: `${riskPct}%`, color: parseFloat(riskPct) > 2 ? 'var(--red)' : 'var(--green)' },
          { label: 'Trades / Day', value: tradesPerDay, color: 'var(--text-primary)' },
          { label: 'Expected Daily', value: `$${expectedDaily}`, color: 'var(--green)' },
          { label: 'Expected Monthly', value: `$${expectedMonthly}`, color: 'var(--green)' },
          { label: 'Worst Drawdown', value: `$${worstDrawdown}`, color: 'var(--red)' },
          { label: 'Recovery Factor', value: (kpis.totalR / Math.abs(kpis.maxDD)).toFixed(1), color: 'var(--text-primary)' },
        ].map((item) => (
          <div key={item.label} className="px-2 py-1.5 rounded-lg" style={{ background: 'var(--bg-input)' }}>
            <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{item.label}</p>
            <p className="text-sm font-semibold" style={{ color: item.color }}>{item.value}</p>
          </div>
        ))}
      </div>
    </Card>
  );
}

/** Confluence Picker Modal */
function ConfluencePickerModal({
  isOpen,
  onClose,
  selected,
  onToggle,
}: {
  isOpen: boolean;
  onClose: () => void;
  selected: Set<string>;
  onToggle: (id: string) => void;
}) {
  const [search, setSearch] = useState('');

  const filtered = useMemo(() => {
    if (!search) return ALL_CONFLUENCE;
    const q = search.toLowerCase();
    return ALL_CONFLUENCE.filter(
      (c) => c.label.toLowerCase().includes(q) || c.id.toLowerCase().includes(q)
    );
  }, [search]);

  const grouped = useMemo(() => {
    const map: Record<string, ConfluenceCondition[]> = {};
    for (const c of filtered) {
      if (!map[c.pack]) map[c.pack] = [];
      map[c.pack].push(c);
    }
    return map;
  }, [filtered]);

  return (
    <Modal title="Confluence Conditions" isOpen={isOpen} onClose={onClose} width="480px">
      <input
        type="text"
        className="w-full px-3 py-2 rounded-lg text-sm mb-3"
        style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        placeholder="Search conditions..."
      />
      <div className="space-y-0.5 overflow-y-auto" style={{ maxHeight: 400 }}>
        {Object.entries(grouped).map(([pack, conds]) => (
          <div key={pack}>
            <div
              className="text-[10px] font-semibold uppercase tracking-wider px-2 py-1.5 sticky top-0"
              style={{ color: 'var(--text-muted)', background: 'var(--bg-secondary)' }}
            >
              {pack}
            </div>
            {conds.map((c) => {
              const active = selected.has(c.id);
              return (
                <button
                  key={c.id}
                  className="w-full text-left px-3 py-2 rounded-lg text-sm flex items-center gap-2 transition-colors"
                  style={{
                    background: active ? 'var(--accent-muted)' : 'transparent',
                    color: active ? 'var(--accent)' : 'var(--text-secondary)',
                  }}
                  onClick={() => onToggle(c.id)}
                >
                  <span
                    className="w-4 h-4 rounded border flex items-center justify-center text-[10px]"
                    style={{
                      borderColor: active ? 'var(--accent)' : 'var(--border)',
                      background: active ? 'var(--accent)' : 'transparent',
                      color: active ? '#000' : 'transparent',
                    }}
                  >
                    {active ? '\u2713' : ''}
                  </span>
                  <span className="flex-1">{c.label}</span>
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{c.id}</span>
                </button>
              );
            })}
          </div>
        ))}
      </div>
    </Modal>
  );
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function StrategyBuilderV4() {
  // -- State --
  const [ticker, setTicker] = useState('SPY');
  const [direction, setDirection] = useState('LONG');
  const [timeframe, setTimeframe] = useState('5Min');
  const [session, setSession] = useState('RTH');
  const [lookbackDays, setLookbackDays] = useState(30);
  const [selectedTriggerId, setSelectedTriggerId] = useState('ema-price-cross-up');
  const [confluenceIds, setConfluenceIds] = useState<Set<string>>(new Set(['5M-EMA_STACK-BULL', '1D-MACD_LINE-BULL']));
  const [stopMethod, setStopMethod] = useState('Swing');
  const [targetMethod, setTargetMethod] = useState('R:R');
  const [targetRR, setTargetRR] = useState(2.0);
  const [showConfluenceModal, setShowConfluenceModal] = useState(false);
  const [showTriggerModal, setShowTriggerModal] = useState(false);
  const [showPresetModal, setShowPresetModal] = useState(false);
  const [backtestRan, setBacktestRan] = useState(true);
  const [triggerSearch, setTriggerSearch] = useState('');
  const [activePreset, setActivePreset] = useState<string | null>(null);

  const selectedTrigger = MOCK_TRIGGERS.find(t => t.id === selectedTriggerId);

  // -- Pipeline nodes --
  const pipelineNodes: PipelineNode[] = useMemo(() => [
    {
      id: 'data',
      label: 'Data Feed',
      sublabel: `${ticker} | ${timeframe}`,
      status: 'configured' as const,
      icon: '\u{1F4CA}',
    },
    {
      id: 'indicators',
      label: 'Indicators',
      sublabel: selectedTrigger ? selectedTrigger.pack : 'Select pack',
      status: selectedTrigger ? 'configured' as const : 'empty' as const,
      icon: '\u{1F4C8}',
    },
    {
      id: 'confluence',
      label: 'Confluence',
      sublabel: `${confluenceIds.size} conditions`,
      status: confluenceIds.size > 0 ? 'configured' as const : 'empty' as const,
      icon: '\u{1F517}',
    },
    {
      id: 'entry',
      label: 'Entry',
      sublabel: selectedTrigger ? `[${selectedTrigger.execType}]` : 'No trigger',
      status: selectedTrigger ? 'active' as const : 'empty' as const,
      icon: '\u{1F3AF}',
    },
    {
      id: 'position',
      label: 'Position',
      sublabel: `${stopMethod} / ${targetMethod}`,
      status: 'configured' as const,
      icon: '\u{1F4B0}',
    },
    {
      id: 'exit',
      label: 'Exit',
      sublabel: targetMethod !== 'None' ? `${targetRR}R target` : 'Signal only',
      status: 'configured' as const,
      icon: '\u{1F6AA}',
    },
  ], [ticker, timeframe, selectedTrigger, confluenceIds.size, stopMethod, targetMethod, targetRR]);

  // -- Strategy DNA calculation (mock) --
  const strategyDNA = useMemo(() => ({
    aggression: timeframe === '1Min' ? 85 : timeframe === '5Min' ? 65 : timeframe === '15Min' ? 45 : 25,
    selectivity: Math.min(100, confluenceIds.size * 25 + 15),
    speed: timeframe === '1Min' ? 90 : timeframe === '5Min' ? 70 : timeframe === '15Min' ? 50 : 30,
    consistency: Math.min(100, MOCK_KPIS.rSquared * 100),
    risk: stopMethod !== 'None' ? 80 : 30,
    trend: confluenceIds.has('1H-EMA_STACK-BULL') || confluenceIds.has('1D-MACD_LINE-BULL') ? 85 : 40,
  }), [timeframe, confluenceIds, stopMethod]);

  // -- Confidence score (mock) --
  const confidenceScore = useMemo(() => {
    let score = 0;
    if (MOCK_KPIS.totalTrades > 100) score += 25;
    else if (MOCK_KPIS.totalTrades > 50) score += 15;
    if (MOCK_KPIS.winRate > 55) score += 20;
    if (MOCK_KPIS.profitFactor > 1.5) score += 20;
    if (MOCK_KPIS.rSquared > 0.8) score += 15;
    if (confluenceIds.size >= 2) score += 10;
    if (stopMethod !== 'None') score += 10;
    return Math.min(100, score);
  }, [confluenceIds.size, stopMethod]);

  // -- Mock sparkline data for indicator preview --
  const sparklineData = useMemo(() => ({
    ema: [44, 45, 44.5, 46, 47, 46.5, 48, 49, 48.5, 50, 51, 50.5, 52, 53, 52],
    macd: [0.2, 0.4, 0.3, 0.5, 0.8, 0.6, 0.9, 1.1, 0.8, 1.2, 1.4, 1.1, 1.5, 1.3, 1.6],
    rsi: [35, 42, 55, 48, 62, 58, 65, 70, 55, 68, 72, 60, 65, 58, 63],
    price: [565, 567, 566, 569, 571, 568, 572, 574, 570, 573, 575, 571, 574, 572, 576],
  }), []);

  // -- Handlers --
  const handleToggleConfluence = useCallback((id: string) => {
    setConfluenceIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const handleNodeClick = useCallback((id: string) => {
    if (id === 'entry') setShowTriggerModal(true);
    if (id === 'confluence') setShowConfluenceModal(true);
  }, []);

  const handleApplyPreset = useCallback((preset: typeof STRATEGY_PRESETS[0]) => {
    setTicker(preset.ticker);
    setDirection(preset.direction);
    setTimeframe(preset.timeframe);
    setSelectedTriggerId(preset.trigger);
    setConfluenceIds(new Set(preset.confluences));
    setStopMethod(preset.stop);
    setTargetMethod(preset.target);
    setActivePreset(preset.name);
    setShowPresetModal(false);
    setBacktestRan(true);
  }, []);

  // -- Filtered triggers for search --
  const filteredTriggers = useMemo(() => {
    if (!triggerSearch) return MOCK_TRIGGERS;
    const q = triggerSearch.toLowerCase();
    return MOCK_TRIGGERS.filter(t => t.name.toLowerCase().includes(q) || t.pack.toLowerCase().includes(q));
  }, [triggerSearch]);

  const triggersByPack = useMemo(() => {
    const map: Record<string, TriggerDef[]> = {};
    for (const t of filteredTriggers) {
      if (!map[t.pack]) map[t.pack] = [];
      map[t.pack].push(t);
    }
    return map;
  }, [filteredTriggers]);

  // -------------------------------------------------------------------------
  // RENDER
  // -------------------------------------------------------------------------
  return (
    <div style={{ animation: 'fade-in-up 0.3s ease-out' }}>
      <style>{ANIMATION_STYLES}</style>

      {/* ============ HEADER ============ */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <h1 className="text-2xl font-bold mb-1">Strategy Builder</h1>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
            Design, backtest, and validate trading strategies
          </p>
        </div>
        <div className="flex gap-2">
          <button
            className="px-4 py-2 rounded-lg text-xs font-medium"
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
            onClick={() => setShowPresetModal(true)}
          >
            Quick Presets
          </button>
          <button
            className="px-4 py-2 rounded-lg text-xs font-medium"
            style={{ background: 'var(--accent)', color: '#000' }}
            onClick={() => setBacktestRan(true)}
          >
            Run Backtest
          </button>
        </div>
      </div>

      {/* ============ PIPELINE GRAPH ============ */}
      <Card className="mb-4">
        <div className="flex items-center gap-2 mb-2">
          <h3 className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>Strategy Pipeline</h3>
          {activePreset && (
            <span
              className="text-[10px] px-2 py-0.5 rounded-full"
              style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
            >
              {activePreset}
            </span>
          )}
        </div>
        <PipelineGraph nodes={pipelineNodes} onNodeClick={handleNodeClick} />
      </Card>

      {/* ============ MAIN CONTENT: 3 COLUMNS ============ */}
      <div className="grid grid-cols-12 gap-4">
        {/* -- LEFT: Configuration (4 cols) -- */}
        <div className="col-span-4 space-y-4">
          {/* Config Card */}
          <Card>
            <h3 className="text-xs font-semibold mb-3" style={{ color: 'var(--text-muted)' }}>Configuration</h3>
            <div className="space-y-3">
              <div>
                <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>Ticker</label>
                <input
                  type="text"
                  className="w-full px-3 py-2 rounded-lg text-sm font-mono"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                  value={ticker}
                  onChange={(e) => setTicker(e.target.value.toUpperCase())}
                />
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>Direction</label>
                  <select
                    className="w-full px-2 py-2 rounded-lg text-sm"
                    style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                    value={direction}
                    onChange={(e) => setDirection(e.target.value)}
                  >
                    {DIRECTIONS.map(d => <option key={d} value={d}>{d}</option>)}
                  </select>
                </div>
                <div>
                  <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>Timeframe</label>
                  <select
                    className="w-full px-2 py-2 rounded-lg text-sm"
                    style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                    value={timeframe}
                    onChange={(e) => setTimeframe(e.target.value)}
                  >
                    {TIMEFRAMES.map(t => <option key={t} value={t}>{t}</option>)}
                  </select>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>Session</label>
                  <select
                    className="w-full px-2 py-2 rounded-lg text-sm"
                    style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                    value={session}
                    onChange={(e) => setSession(e.target.value)}
                  >
                    {SESSIONS.map(s => <option key={s} value={s}>{s}</option>)}
                  </select>
                </div>
                <div>
                  <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>Lookback</label>
                  <div className="flex items-center gap-1">
                    <input
                      type="number"
                      className="w-full px-2 py-2 rounded-lg text-sm"
                      style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                      value={lookbackDays}
                      onChange={(e) => setLookbackDays(parseInt(e.target.value) || 0)}
                    />
                    <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>days</span>
                  </div>
                </div>
              </div>
            </div>
          </Card>

          {/* Entry Trigger */}
          <Card>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>Entry Trigger</h3>
              <button
                className="text-[10px] px-2 py-0.5 rounded"
                style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
                onClick={() => setShowTriggerModal(true)}
              >
                Change
              </button>
            </div>
            {selectedTrigger && (
              <div
                className="flex items-center gap-2 px-3 py-2.5 rounded-lg"
                style={{
                  background: 'var(--bg-input)',
                  border: '1px solid var(--border)',
                  animation: 'glow-pulse 3s ease-in-out infinite',
                }}
              >
                <ExecBadge type={selectedTrigger.execType} />
                <div className="flex-1">
                  <p className="text-sm font-medium">{selectedTrigger.name}</p>
                  <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{selectedTrigger.pack}</p>
                </div>
              </div>
            )}
          </Card>

          {/* Confluence */}
          <Card>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>
                Confluence ({confluenceIds.size})
              </h3>
              <button
                className="text-[10px] px-2 py-0.5 rounded"
                style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
                onClick={() => setShowConfluenceModal(true)}
              >
                + Add
              </button>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {Array.from(confluenceIds).map(id => {
                const cond = ALL_CONFLUENCE.find(c => c.id === id);
                return (
                  <span
                    key={id}
                    className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[10px]"
                    style={{
                      background: 'color-mix(in srgb, var(--accent) 12%, transparent)',
                      border: '1px solid color-mix(in srgb, var(--accent) 30%, transparent)',
                      color: 'var(--accent)',
                    }}
                  >
                    {cond?.label || id}
                    <button
                      onClick={() => handleToggleConfluence(id)}
                      className="hover:opacity-70 ml-0.5"
                      style={{ color: 'var(--text-muted)' }}
                    >
                      x
                    </button>
                  </span>
                );
              })}
              {confluenceIds.size === 0 && (
                <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>No conditions selected</p>
              )}
            </div>
          </Card>

          {/* Exit Config */}
          <Card>
            <h3 className="text-xs font-semibold mb-3" style={{ color: 'var(--text-muted)' }}>Exit Strategy</h3>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>Stop</label>
                <select
                  className="w-full px-2 py-1.5 rounded-lg text-sm"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                  value={stopMethod}
                  onChange={(e) => setStopMethod(e.target.value)}
                >
                  {STOP_METHODS.map(s => <option key={s} value={s}>{s}</option>)}
                </select>
              </div>
              <div>
                <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>Target</label>
                <select
                  className="w-full px-2 py-1.5 rounded-lg text-sm"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                  value={targetMethod}
                  onChange={(e) => setTargetMethod(e.target.value)}
                >
                  {TARGET_METHODS.map(t => <option key={t} value={t}>{t}</option>)}
                </select>
              </div>
            </div>
            {targetMethod === 'R:R' && (
              <div className="mt-2">
                <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>R:R Ratio</label>
                <input
                  type="number"
                  className="w-full px-2 py-1.5 rounded-lg text-sm"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                  value={targetRR}
                  step={0.5}
                  min={0.5}
                  onChange={(e) => setTargetRR(parseFloat(e.target.value) || 1)}
                />
              </div>
            )}
            <div
              className="mt-3 px-2.5 py-2 rounded-lg text-xs"
              style={{ background: 'var(--bg-input)', color: 'var(--text-secondary)' }}
            >
              {stopMethod} stop | {targetMethod === 'None' ? 'Signal exit only' : `${targetMethod} ${targetMethod === 'R:R' ? `(${targetRR}R)` : ''} target`} | Opposite signal exit
            </div>
          </Card>

          {/* Live Indicator Preview */}
          <Card>
            <h3 className="text-xs font-semibold mb-3" style={{ color: 'var(--text-muted)' }}>Indicator Preview</h3>
            <div className="space-y-2">
              <IndicatorSparkline data={sparklineData.price} color="var(--text-secondary)" label="Price" />
              <IndicatorSparkline data={sparklineData.ema} color="var(--blue)" label="EMA" />
              <IndicatorSparkline data={sparklineData.macd} color="var(--orange)" label="MACD" />
              <IndicatorSparkline data={sparklineData.rsi} color="var(--accent)" label="RSI" />
            </div>
            <p className="text-[10px] mt-2" style={{ color: 'var(--text-muted)' }}>
              Live preview of {ticker} {timeframe} indicators
            </p>
          </Card>
        </div>

        {/* -- CENTER: Results (5 cols) -- */}
        <div className="col-span-5 space-y-4">
          {backtestRan && (
            <>
              {/* KPI Strip */}
              <div className="grid grid-cols-4 gap-2">
                <MetricCard label="Win Rate" value={`${MOCK_KPIS.winRate}%`} delta={MOCK_KPIS.winRate > 55 ? 'Above avg' : 'Below avg'} positive={MOCK_KPIS.winRate > 55} />
                <MetricCard label="Profit Factor" value={MOCK_KPIS.profitFactor.toFixed(2)} delta={MOCK_KPIS.profitFactor > 1.5 ? 'Profitable' : 'Marginal'} positive={MOCK_KPIS.profitFactor > 1.5} />
                <MetricCard label="Daily R" value={`+${MOCK_KPIS.dailyR.toFixed(2)}`} delta={`${MOCK_KPIS.totalTrades} trades`} positive />
                <MetricCard label="R-Squared" value={MOCK_KPIS.rSquared.toFixed(2)} delta={MOCK_KPIS.rSquared > 0.8 ? 'Consistent' : 'Erratic'} positive={MOCK_KPIS.rSquared > 0.8} />
              </div>

              {/* Interactive Equity Curve */}
              <Card>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>
                    Equity Curve — {MOCK_KPIS.totalR.toFixed(1)}R total
                  </h3>
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Hover dots for trade details</span>
                </div>
                <InteractiveEquityCurve trades={MOCK_TRADES} width={560} height={180} />
              </Card>

              {/* Price Chart Placeholder */}
              <Card>
                <h3 className="text-xs font-semibold mb-2" style={{ color: 'var(--text-muted)' }}>
                  Price Chart — {ticker} {timeframe}
                </h3>
                <div
                  className="rounded-lg flex items-center justify-center relative overflow-hidden"
                  style={{ background: 'var(--bg-input)', height: 280 }}
                >
                  {/* Mock candlestick pattern */}
                  <svg width="100%" height="100%" viewBox="0 0 500 260" preserveAspectRatio="none">
                    {/* Grid */}
                    {[0.2, 0.4, 0.6, 0.8].map(pct => (
                      <line key={pct} x1="0" y1={pct * 260} x2="500" y2={pct * 260} stroke="var(--border)" strokeWidth="0.5" opacity="0.3" />
                    ))}
                    {/* Mock candles */}
                    {Array.from({ length: 40 }, (_, i) => {
                      const x = 10 + i * 12;
                      const baseY = 130 + Math.sin(i * 0.3) * 40 + Math.cos(i * 0.15) * 20;
                      const bodyH = 5 + Math.random() * 15;
                      const wickH = 3 + Math.random() * 8;
                      const bullish = Math.random() > 0.42;
                      return (
                        <g key={i}>
                          <line x1={x + 3} y1={baseY - wickH - bodyH / 2} x2={x + 3} y2={baseY + wickH + bodyH / 2}
                            stroke={bullish ? 'var(--green)' : 'var(--red)'} strokeWidth="1" opacity="0.6" />
                          <rect x={x} y={baseY - bodyH / 2} width="6" height={bodyH} rx="1"
                            fill={bullish ? 'var(--green)' : 'var(--red)'} opacity="0.8" />
                        </g>
                      );
                    })}
                    {/* EMA lines */}
                    <path
                      d={Array.from({ length: 40 }, (_, i) => {
                        const x = 13 + i * 12;
                        const y = 130 + Math.sin(i * 0.3) * 35 + Math.cos(i * 0.15) * 18;
                        return `${i === 0 ? 'M' : 'L'}${x},${y}`;
                      }).join(' ')}
                      fill="none" stroke="var(--blue)" strokeWidth="1.5" opacity="0.7"
                    />
                    <path
                      d={Array.from({ length: 40 }, (_, i) => {
                        const x = 13 + i * 12;
                        const y = 140 + Math.sin(i * 0.25) * 30 + Math.cos(i * 0.12) * 15;
                        return `${i === 0 ? 'M' : 'L'}${x},${y}`;
                      }).join(' ')}
                      fill="none" stroke="var(--orange)" strokeWidth="1.5" opacity="0.5"
                    />
                    {/* Entry markers */}
                    {[8, 18, 28].map(i => {
                      const x = 13 + i * 12;
                      const y = 125 + Math.sin(i * 0.3) * 40 + Math.cos(i * 0.15) * 20;
                      return (
                        <g key={`entry-${i}`}>
                          <polygon points={`${x},${y + 15} ${x - 5},${y + 22} ${x + 5},${y + 22}`} fill="var(--green)" opacity="0.9" />
                        </g>
                      );
                    })}
                  </svg>
                  <span
                    className="absolute bottom-2 right-3 text-[10px]"
                    style={{ color: 'var(--text-muted)' }}
                  >
                    OHLC + Indicators + Trade Markers
                  </span>
                </div>
              </Card>

              {/* Trade History */}
              <Card>
                <h3 className="text-xs font-semibold mb-3" style={{ color: 'var(--text-muted)' }}>
                  Trade History ({MOCK_TRADES.length} trades)
                </h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr style={{ color: 'var(--text-muted)' }}>
                        {['#', 'Entry', 'Exit', 'Price', 'R', 'Type', 'Reason'].map(h => (
                          <th key={h} className="text-left py-1.5 px-1 font-medium">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {MOCK_TRADES.slice(0, 8).map(trade => (
                        <tr key={trade.id} className="border-t" style={{ borderColor: 'var(--border)' }}>
                          <td className="py-1.5 px-1" style={{ color: 'var(--text-muted)' }}>{trade.id}</td>
                          <td className="py-1.5 px-1" style={{ color: 'var(--text-secondary)' }}>{trade.entryTime}</td>
                          <td className="py-1.5 px-1" style={{ color: 'var(--text-secondary)' }}>{trade.exitTime}</td>
                          <td className="py-1.5 px-1 font-mono" style={{ color: 'var(--text-secondary)' }}>
                            {trade.entryPrice.toFixed(2)}
                          </td>
                          <td className="py-1.5 px-1 font-mono font-semibold" style={{
                            color: trade.rMultiple >= 0 ? 'var(--green)' : 'var(--red)',
                          }}>
                            {trade.rMultiple >= 0 ? '+' : ''}{trade.rMultiple.toFixed(2)}
                          </td>
                          <td className="py-1.5 px-1"><ExecBadge type={trade.execType} /></td>
                          <td className="py-1.5 px-1">
                            <span
                              className="text-[10px] px-1.5 py-0.5 rounded"
                              style={{
                                color: EXIT_REASON_COLORS[trade.exitReason] || 'var(--text-muted)',
                                background: `color-mix(in srgb, ${EXIT_REASON_COLORS[trade.exitReason] || 'var(--text-muted)'} 12%, transparent)`,
                              }}
                            >
                              {trade.exitReason}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            </>
          )}

          {!backtestRan && (
            <Card>
              <div className="flex flex-col items-center justify-center py-16">
                <div
                  className="w-16 h-16 rounded-2xl flex items-center justify-center text-2xl mb-4"
                  style={{ background: 'var(--accent-muted)' }}
                >
                  {'\u{1F680}'}
                </div>
                <h3 className="text-lg font-semibold mb-2">Ready to Backtest</h3>
                <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
                  Configure your strategy pipeline and hit Run Backtest
                </p>
                <button
                  className="px-6 py-2.5 rounded-lg text-sm font-medium"
                  style={{ background: 'var(--accent)', color: '#000' }}
                  onClick={() => setBacktestRan(true)}
                >
                  Run Backtest
                </button>
              </div>
            </Card>
          )}
        </div>

        {/* -- RIGHT: DNA + Risk + AI (3 cols) -- */}
        <div className="col-span-3 space-y-4">
          {/* Strategy DNA */}
          <Card>
            <h3 className="text-xs font-semibold mb-2" style={{ color: 'var(--text-muted)' }}>Strategy DNA</h3>
            <div className="flex justify-center">
              <StrategyDNAChart values={strategyDNA} size={200} />
            </div>
            <div className="grid grid-cols-2 gap-1 mt-2">
              {DNA_AXES.map(axis => (
                <div key={axis.key} className="flex items-center gap-1.5 text-[10px]">
                  <div
                    className="w-1.5 h-1.5 rounded-full"
                    style={{
                      background: (strategyDNA as Record<string, number>)[axis.key] > 60 ? 'var(--green)' : 'var(--text-muted)',
                    }}
                  />
                  <span style={{ color: 'var(--text-muted)' }}>{axis.label}</span>
                  <span className="font-mono ml-auto" style={{ color: 'var(--text-secondary)' }}>
                    {(strategyDNA as Record<string, number>)[axis.key]}
                  </span>
                </div>
              ))}
            </div>
          </Card>

          {/* Confidence Gauge */}
          {backtestRan && (
            <Card>
              <h3 className="text-xs font-semibold mb-2" style={{ color: 'var(--text-muted)' }}>Confidence Score</h3>
              <div className="flex justify-center">
                <ConfidenceGauge score={confidenceScore} />
              </div>
              <div className="space-y-1 mt-3">
                {[
                  { label: 'Trade count > 100', met: MOCK_KPIS.totalTrades > 100 },
                  { label: 'Win rate > 55%', met: MOCK_KPIS.winRate > 55 },
                  { label: 'Profit factor > 1.5', met: MOCK_KPIS.profitFactor > 1.5 },
                  { label: 'R-squared > 0.8', met: MOCK_KPIS.rSquared > 0.8 },
                  { label: '2+ confluences', met: confluenceIds.size >= 2 },
                ].map(check => (
                  <div key={check.label} className="flex items-center gap-1.5 text-[10px]">
                    <span style={{ color: check.met ? 'var(--green)' : 'var(--red)' }}>
                      {check.met ? '\u2713' : '\u2717'}
                    </span>
                    <span style={{ color: check.met ? 'var(--text-secondary)' : 'var(--text-muted)' }}>
                      {check.label}
                    </span>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* Risk Calculator */}
          {backtestRan && <RiskCalculator kpis={MOCK_KPIS} />}

          {/* AI Insight */}
          <AIInsightCard ticker={ticker} direction={direction} />

          {/* Save Action */}
          {backtestRan && (
            <button
              className="w-full py-3 rounded-lg text-sm font-medium"
              style={{
                background: 'var(--accent)',
                color: '#000',
                boxShadow: '0 0 20px rgba(0,255,136,0.2)',
              }}
            >
              Save Strategy
            </button>
          )}
        </div>
      </div>

      {/* ============ MODALS ============ */}

      {/* Trigger Picker Modal */}
      <Modal
        title="Select Entry Trigger"
        isOpen={showTriggerModal}
        onClose={() => setShowTriggerModal(false)}
        width="500px"
      >
        <input
          type="text"
          className="w-full px-3 py-2 rounded-lg text-sm mb-3"
          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
          value={triggerSearch}
          onChange={(e) => setTriggerSearch(e.target.value)}
          placeholder="Search triggers..."
        />
        <div className="space-y-0.5 overflow-y-auto" style={{ maxHeight: 400 }}>
          {Object.entries(triggersByPack).map(([pack, triggers]) => (
            <div key={pack}>
              <div
                className="text-[10px] font-semibold uppercase tracking-wider px-2 py-1.5 sticky top-0"
                style={{ color: 'var(--text-muted)', background: 'var(--bg-secondary)' }}
              >
                {pack}
              </div>
              {triggers.map(t => (
                <button
                  key={t.id}
                  className="w-full text-left px-3 py-2 rounded-lg text-sm flex items-center gap-2 transition-colors"
                  style={{
                    background: selectedTriggerId === t.id ? 'var(--accent-muted)' : 'transparent',
                    color: selectedTriggerId === t.id ? 'var(--accent)' : 'var(--text-secondary)',
                  }}
                  onClick={() => {
                    setSelectedTriggerId(t.id);
                    setShowTriggerModal(false);
                  }}
                >
                  <ExecBadge type={t.execType} />
                  <span className="flex-1">{t.name}</span>
                </button>
              ))}
            </div>
          ))}
        </div>
      </Modal>

      {/* Confluence Picker Modal */}
      <ConfluencePickerModal
        isOpen={showConfluenceModal}
        onClose={() => setShowConfluenceModal(false)}
        selected={confluenceIds}
        onToggle={handleToggleConfluence}
      />

      {/* Preset Modal */}
      <Modal
        title="Quick Presets"
        isOpen={showPresetModal}
        onClose={() => setShowPresetModal(false)}
        width="600px"
      >
        <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
          One-click strategy templates. Select a preset to auto-fill all configuration.
        </p>
        <div className="space-y-3">
          {STRATEGY_PRESETS.map(preset => (
            <button
              key={preset.name}
              className="w-full text-left rounded-xl p-4 transition-all"
              style={{
                background: 'var(--bg-input)',
                border: activePreset === preset.name ? '2px solid var(--accent)' : '1px solid var(--border)',
              }}
              onClick={() => handleApplyPreset(preset)}
            >
              <div className="flex items-center justify-between mb-1">
                <h4 className="text-sm font-semibold">{preset.name}</h4>
                <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>
                  {preset.ticker} | {preset.direction} | {preset.timeframe}
                </span>
              </div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{preset.description}</p>
              <div className="flex flex-wrap gap-1 mt-2">
                {preset.confluences.map(c => {
                  const cond = ALL_CONFLUENCE.find(x => x.id === c);
                  return (
                    <span
                      key={c}
                      className="text-[10px] px-1.5 py-0.5 rounded"
                      style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
                    >
                      {cond?.label || c}
                    </span>
                  );
                })}
              </div>
            </button>
          ))}
        </div>
      </Modal>
    </div>
  );
}
