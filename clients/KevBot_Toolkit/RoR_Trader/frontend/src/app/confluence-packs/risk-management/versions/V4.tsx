'use client';

import { useState, useCallback, useMemo } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import Modal from '@/components/Modal';
import MetricCard from '@/components/MetricCard';

/* ─────────────────────────── Types ─────────────────────────── */

interface PackParam {
  key: string;
  label: string;
  type: 'int' | 'float' | 'select';
  value: number | string;
  min?: number;
  max?: number;
  step?: number;
  help?: string;
}

interface RiskPack {
  id: string;
  name: string;
  category: 'Volatility' | 'Fixed' | 'Structure' | 'Trailing';
  stopType: string;
  enabled: boolean;
  isDefault: boolean;
  params: PackParam[];
  stopSummary: string;
  targetSummary: string;
  contextualHelp: string[];
}

/* ─────────────────────────── Mock Data ─────────────────────────── */

const initialPacks: RiskPack[] = [
  {
    id: 'atr-default',
    name: 'ATR-Based (Default)',
    category: 'Volatility',
    stopType: 'ATR',
    enabled: true,
    isDefault: true,
    params: [
      { key: 'stop_atr_mult', label: 'Stop ATR Mult', type: 'float', value: 1.5, min: 0.5, max: 5.0, step: 0.1, help: 'ATR multiplier for stop distance from entry' },
      { key: 'target_atr_mult', label: 'Target ATR Mult', type: 'float', value: 3.0, min: 0.0, max: 10.0, step: 0.1, help: 'Set to 0 for no ATR target' },
    ],
    stopSummary: 'ATR x1.5',
    targetSummary: 'ATR x3.0',
    contextualHelp: [
      'ATR measures volatility over the last 14 bars by default.',
      'Higher multipliers give more room but increase risk per trade.',
      'ATR stops automatically adapt to market conditions.',
    ],
  },
  {
    id: 'atr-tight',
    name: 'ATR-Based (Tight)',
    category: 'Volatility',
    stopType: 'ATR',
    enabled: true,
    isDefault: false,
    params: [
      { key: 'stop_atr_mult', label: 'Stop ATR Mult', type: 'float', value: 1.0, min: 0.5, max: 5.0, step: 0.1, help: 'ATR multiplier for stop distance from entry' },
      { key: 'target_atr_mult', label: 'Target ATR Mult', type: 'float', value: 2.0, min: 0.0, max: 10.0, step: 0.1, help: 'Set to 0 for no ATR target' },
    ],
    stopSummary: 'ATR x1.0',
    targetSummary: 'ATR x2.0',
    contextualHelp: [
      'Tighter ATR stops for scalping or low-volatility setups.',
      '1:2 risk-reward provides a more aggressive profile.',
    ],
  },
  {
    id: 'fixed-default',
    name: 'Fixed Dollar',
    category: 'Fixed',
    stopType: 'Fixed $',
    enabled: true,
    isDefault: true,
    params: [
      { key: 'stop_amount', label: 'Stop ($)', type: 'float', value: 1.0, min: 0.01, max: 100.0, step: 0.01, help: 'Fixed dollar distance from entry to stop' },
      { key: 'target_amount', label: 'Target ($)', type: 'float', value: 2.0, min: 0.0, max: 100.0, step: 0.01, help: 'Fixed dollar distance to target (0 = no target)' },
    ],
    stopSummary: '$1.00',
    targetSummary: '$2.00',
    contextualHelp: [
      'Fixed dollar stops use a constant distance regardless of volatility.',
      'Best for instruments with stable price ranges (e.g., SPY 1-min scalps).',
    ],
  },
  {
    id: 'pct-default',
    name: 'Percentage',
    category: 'Fixed',
    stopType: 'Pct',
    enabled: false,
    isDefault: true,
    params: [
      { key: 'stop_pct', label: 'Stop (%)', type: 'float', value: 0.5, min: 0.01, max: 10.0, step: 0.01, help: 'Percentage distance from entry to stop' },
      { key: 'target_pct', label: 'Target (%)', type: 'float', value: 1.0, min: 0.0, max: 20.0, step: 0.01, help: 'Percentage distance to target (0 = no target)' },
    ],
    stopSummary: '0.5%',
    targetSummary: '1.0%',
    contextualHelp: [
      'Percentage stops scale with instrument price.',
      'Works well across different price levels without manual adjustment.',
    ],
  },
  {
    id: 'swing-default',
    name: 'Swing Stop',
    category: 'Structure',
    stopType: 'Swing',
    enabled: true,
    isDefault: true,
    params: [
      { key: 'lookback', label: 'Lookback', type: 'int', value: 5, min: 2, max: 50, help: 'Bars to search for swing high/low' },
      { key: 'padding', label: 'Padding ($)', type: 'float', value: 0.05, min: 0.0, max: 10.0, step: 0.01, help: 'Buffer beyond the swing point' },
      { key: 'rr_ratio', label: 'R:R Ratio', type: 'float', value: 2.0, min: 0.0, max: 10.0, step: 0.1, help: 'Target as a multiple of stop distance (0 = no target)' },
    ],
    stopSummary: 'Swing (5 bar, $0.05 pad)',
    targetSummary: '2.0R',
    contextualHelp: [
      'Places the stop at the recent swing low (longs) or swing high (shorts).',
      'Respects market structure at logical support/resistance levels.',
    ],
  },
  {
    id: 'atr-trailing',
    name: 'ATR Trailing',
    category: 'Trailing',
    stopType: 'Trail',
    enabled: false,
    isDefault: true,
    params: [
      { key: 'stop_atr_mult', label: 'Initial Stop ATR', type: 'float', value: 1.5, min: 0.5, max: 5.0, step: 0.1, help: 'Initial stop distance from entry' },
      { key: 'trail_atr_mult', label: 'Trail ATR', type: 'float', value: 1.0, min: 0.3, max: 5.0, step: 0.1, help: 'Trailing stop ATR distance' },
      { key: 'trail_activation_r', label: 'Activation (R)', type: 'float', value: 0.5, min: 0.0, max: 5.0, step: 0.1, help: 'Profit in R-multiples before trailing begins' },
      { key: 'target_atr_mult', label: 'Target ATR', type: 'float', value: 0.0, min: 0.0, max: 10.0, step: 0.1, help: 'Optional fixed ATR target; 0 = let the trail ride' },
    ],
    stopSummary: 'ATR x1.5 -> Trail x1.0',
    targetSummary: 'None (trail)',
    contextualHelp: [
      'Starts with a standard ATR stop, then transitions to trailing.',
      'The trailing stop ratchets in your favor -- never moves backward.',
    ],
  },
  {
    id: 'breakeven-default',
    name: 'Breakeven Stop',
    category: 'Trailing',
    stopType: 'BE',
    enabled: false,
    isDefault: true,
    params: [
      { key: 'stop_atr_mult', label: 'Initial Stop ATR', type: 'float', value: 1.5, min: 0.5, max: 5.0, step: 0.1, help: 'Initial stop distance from entry' },
      { key: 'be_activation_r', label: 'BE Activation (R)', type: 'float', value: 1.0, min: 0.1, max: 5.0, step: 0.1, help: 'Profit in R before stop moves to breakeven' },
      { key: 'be_offset', label: 'BE Offset ($)', type: 'float', value: 0.0, min: 0.0, max: 2.0, step: 0.01, help: 'Small offset above entry for commissions' },
      { key: 'target_atr_mult', label: 'Target ATR', type: 'float', value: 3.0, min: 0.0, max: 10.0, step: 0.1, help: 'ATR target multiplier; 0 = exit via signal only' },
    ],
    stopSummary: 'ATR x1.5 -> BE at 1.0R',
    targetSummary: 'ATR x3.0',
    contextualHelp: [
      'Standard ATR stop that jumps to breakeven once profit threshold is reached.',
      'Once at breakeven, worst outcome is a scratch trade.',
    ],
  },
];

/* ─────────────────────────── SVG Icons ─────────────────────────── */

function StopPlacementDiagram({ pack, size = 160, height = 100 }: { pack: RiskPack; size?: number; height?: number }) {
  // Mock price chart with stop/target visualization
  const entry = 60;
  const pricePoints = [75, 70, 65, 60, 58, 55, 52, 50, 48, 53, 56, entry, 64, 68, 72, 76, 80, 78, 82, 85];
  const entryIdx = 11;

  // Calculate stop and target positions based on pack type
  let stopLevel = 50;
  let targetLevel = 85;
  let trailPoints: number[] | null = null;

  if (pack.stopType === 'ATR') {
    const mult = Number(pack.params.find((p) => p.key === 'stop_atr_mult')?.value) || 1.5;
    const tMult = Number(pack.params.find((p) => p.key === 'target_atr_mult')?.value) || 3.0;
    const atr = 5; // mock ATR value
    stopLevel = entry - mult * atr;
    targetLevel = tMult > 0 ? entry + tMult * atr : entry + 25;
  } else if (pack.stopType === 'Fixed $') {
    const stopAmt = Number(pack.params.find((p) => p.key === 'stop_amount')?.value) || 1.0;
    const targetAmt = Number(pack.params.find((p) => p.key === 'target_amount')?.value) || 2.0;
    stopLevel = entry - stopAmt * 5;
    targetLevel = targetAmt > 0 ? entry + targetAmt * 5 : entry + 20;
  } else if (pack.stopType === 'Swing') {
    stopLevel = 48; // swing low
    const rr = Number(pack.params.find((p) => p.key === 'rr_ratio')?.value) || 2.0;
    const risk = entry - stopLevel;
    targetLevel = rr > 0 ? entry + risk * rr : entry + 25;
  } else if (pack.stopType === 'Trail') {
    const mult = Number(pack.params.find((p) => p.key === 'stop_atr_mult')?.value) || 1.5;
    const atr = 5;
    stopLevel = entry - mult * atr;
    // Generate trailing stop line
    trailPoints = pricePoints.slice(entryIdx).map((p, i) => {
      if (i < 3) return stopLevel;
      return Math.max(stopLevel, p - 8);
    });
    targetLevel = entry + 30;
  } else if (pack.stopType === 'BE') {
    const mult = Number(pack.params.find((p) => p.key === 'stop_atr_mult')?.value) || 1.5;
    const atr = 5;
    stopLevel = entry - mult * atr;
    targetLevel = entry + 20;
  } else if (pack.stopType === 'Pct') {
    const pct = Number(pack.params.find((p) => p.key === 'stop_pct')?.value) || 0.5;
    const tPct = Number(pack.params.find((p) => p.key === 'target_pct')?.value) || 1.0;
    stopLevel = entry - entry * pct / 10;
    targetLevel = tPct > 0 ? entry + entry * tPct / 10 : entry + 20;
  }

  // Normalize to SVG coords
  const allVals = [...pricePoints, stopLevel, targetLevel];
  const minVal = Math.min(...allVals) - 5;
  const maxVal = Math.max(...allVals) + 5;
  const range = maxVal - minVal;

  const toY = (v: number) => height - 8 - ((v - minVal) / range) * (height - 16);
  const toX = (i: number) => 8 + (i / (pricePoints.length - 1)) * (size - 16);

  const pricePath = pricePoints.map((p, i) => `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toY(p).toFixed(1)}`).join(' ');
  const entryX = toX(entryIdx);
  const entryY = toY(entry);
  const stopY = toY(stopLevel);
  const targetY = toY(targetLevel);

  return (
    <svg width={size} height={height} viewBox={`0 0 ${size} ${height}`} fill="none">
      {/* Target zone fill */}
      <rect x={entryX} y={targetY} width={size - entryX - 8} height={entryY - targetY} fill="var(--green)" opacity="0.06" rx="2" />
      {/* Stop zone fill */}
      <rect x={entryX} y={entryY} width={size - entryX - 8} height={stopY - entryY} fill="var(--red)" opacity="0.06" rx="2" />

      {/* Price line */}
      <path d={pricePath} stroke="var(--text-secondary)" strokeWidth="1.5" strokeLinecap="round" />

      {/* Entry marker */}
      <circle cx={entryX} cy={entryY} r="3.5" fill="var(--accent)" />
      <circle cx={entryX} cy={entryY} r="6" fill="var(--accent)" opacity="0.2" />

      {/* Stop line */}
      <line x1={entryX} y1={stopY} x2={size - 8} y2={stopY} stroke="var(--red)" strokeWidth="1.5" strokeDasharray="4 2" />
      <text x={size - 6} y={stopY + 3} textAnchor="end" fontSize="7" fill="var(--red)" fontWeight="bold">STOP</text>

      {/* Target line */}
      <line x1={entryX} y1={targetY} x2={size - 8} y2={targetY} stroke="var(--green)" strokeWidth="1.5" strokeDasharray="4 2" />
      <text x={size - 6} y={targetY + 3} textAnchor="end" fontSize="7" fill="var(--green)" fontWeight="bold">TGT</text>

      {/* Entry label */}
      <text x={entryX - 2} y={entryY - 6} textAnchor="middle" fontSize="7" fill="var(--accent)" fontWeight="bold">ENTRY</text>

      {/* Trailing stop line (if applicable) */}
      {trailPoints && (
        <path
          d={trailPoints.map((p, i) => `${i === 0 ? 'M' : 'L'}${toX(entryIdx + i).toFixed(1)},${toY(p).toFixed(1)}`).join(' ')}
          stroke="var(--orange)"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeDasharray="3 2"
        />
      )}

      {/* R:R annotation */}
      {(() => {
        const risk = Math.abs(entry - stopLevel);
        const reward = Math.abs(targetLevel - entry);
        const rr = risk > 0 ? (reward / risk).toFixed(1) : '---';
        return (
          <g>
            {/* Risk bracket */}
            <line x1={entryX - 8} y1={entryY} x2={entryX - 8} y2={stopY} stroke="var(--red)" strokeWidth="1" />
            <text x={entryX - 10} y={(entryY + stopY) / 2 + 3} textAnchor="end" fontSize="6" fill="var(--red)">1R</text>
            {/* Reward bracket */}
            <line x1={entryX - 8} y1={entryY} x2={entryX - 8} y2={targetY} stroke="var(--green)" strokeWidth="1" />
            <text x={entryX - 10} y={(entryY + targetY) / 2 + 3} textAnchor="end" fontSize="6" fill="var(--green)">{rr}R</text>
          </g>
        );
      })()}
    </svg>
  );
}

function StopTypeIcon({ stopType, size = 48 }: { stopType: string; size?: number }) {
  if (stopType === 'ATR') {
    return (
      <svg width={size} height={size} viewBox="0 0 48 48" fill="none">
        {/* Wavy ATR envelope */}
        <path d="M4,18 Q12,10 20,18 Q28,26 36,18 Q42,12 46,16" stroke="var(--orange)" strokeWidth="1.5" opacity="0.4" />
        <path d="M4,30 Q12,38 20,30 Q28,22 36,30 Q42,36 46,32" stroke="var(--orange)" strokeWidth="1.5" opacity="0.4" />
        {/* Center price line */}
        <path d="M4,24 Q12,20 20,24 Q28,28 36,24 Q42,22 46,24" stroke="var(--text-secondary)" strokeWidth="2" strokeLinecap="round" />
        {/* ATR bracket */}
        <line x1="24" y1="14" x2="24" y2="34" stroke="var(--orange)" strokeWidth="2" strokeLinecap="round" />
        <line x1="22" y1="14" x2="26" y2="14" stroke="var(--orange)" strokeWidth="1.5" />
        <line x1="22" y1="34" x2="26" y2="34" stroke="var(--orange)" strokeWidth="1.5" />
      </svg>
    );
  }
  if (stopType === 'Fixed $') {
    return (
      <svg width={size} height={size} viewBox="0 0 48 48" fill="none">
        <text x="24" y="30" textAnchor="middle" fontSize="22" fill="var(--blue)" fontWeight="bold">$</text>
        <circle cx="24" cy="24" r="18" stroke="var(--blue)" strokeWidth="1.5" opacity="0.3" />
      </svg>
    );
  }
  if (stopType === 'Pct') {
    return (
      <svg width={size} height={size} viewBox="0 0 48 48" fill="none">
        <text x="24" y="30" textAnchor="middle" fontSize="18" fill="var(--blue)" fontWeight="bold">%</text>
        <circle cx="24" cy="24" r="18" stroke="var(--blue)" strokeWidth="1.5" opacity="0.3" />
      </svg>
    );
  }
  if (stopType === 'Swing') {
    return (
      <svg width={size} height={size} viewBox="0 0 48 48" fill="none">
        {/* Price with swing low */}
        <path d="M4,20 Q10,24 16,18 Q22,12 28,16 Q34,20 40,14 L46,12" stroke="var(--text-secondary)" strokeWidth="1.5" strokeLinecap="round" />
        {/* Swing low marker */}
        <circle cx="16" cy="18" r="3" fill="var(--green)" />
        <line x1="16" y1="21" x2="16" y2="38" stroke="var(--green)" strokeWidth="1" strokeDasharray="3 2" />
        <text x="16" y="43" textAnchor="middle" fontSize="6" fill="var(--green)">SWING</text>
      </svg>
    );
  }
  if (stopType === 'Trail') {
    return (
      <svg width={size} height={size} viewBox="0 0 48 48" fill="none">
        {/* Price going up */}
        <path d="M4,36 Q12,32 20,28 Q28,24 36,18 L44,14" stroke="var(--text-secondary)" strokeWidth="1.5" strokeLinecap="round" />
        {/* Trailing stop */}
        <path d="M4,42 L12,42 L12,38 L20,38 L20,34 L28,34 L28,28 L36,28 L36,24" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" />
        <text x="42" y="28" fontSize="6" fill="var(--accent)">TRAIL</text>
      </svg>
    );
  }
  // BE
  return (
    <svg width={size} height={size} viewBox="0 0 48 48" fill="none">
      <path d="M4,32 Q12,28 20,24 Q28,20 36,16 L44,14" stroke="var(--text-secondary)" strokeWidth="1.5" strokeLinecap="round" />
      {/* Initial stop */}
      <line x1="4" y1="38" x2="20" y2="38" stroke="var(--red)" strokeWidth="1.5" strokeDasharray="3 2" />
      {/* BE level */}
      <line x1="20" y1="32" x2="44" y2="32" stroke="var(--accent)" strokeWidth="2" />
      <text x="44" y="36" fontSize="6" fill="var(--accent)" textAnchor="end">BE</text>
      {/* Entry dot */}
      <circle cx="12" cy="32" r="2.5" fill="var(--accent)" />
    </svg>
  );
}

/* ─────────────────────────── Helpers ─────────────────────────── */

const categoryColors: Record<string, { color: string; bg: string }> = {
  Volatility: { color: 'var(--orange)', bg: 'var(--orange-muted)' },
  Fixed: { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  Structure: { color: 'var(--green)', bg: 'var(--green-muted)' },
  Trailing: { color: 'var(--accent)', bg: 'var(--accent-muted)' },
};

/* ─────────────────────────── R:R Calculator ─────────────────────────── */

function RiskRewardCalculator() {
  const [entryPrice, setEntryPrice] = useState(150.0);
  const [stopPrice, setStopPrice] = useState(148.5);
  const [targetPrice, setTargetPrice] = useState(153.0);
  const [positionSize, setPositionSize] = useState(100);

  const risk = Math.abs(entryPrice - stopPrice);
  const reward = Math.abs(targetPrice - entryPrice);
  const rr = risk > 0 ? (reward / risk).toFixed(2) : '---';
  const dollarRisk = risk * positionSize;
  const dollarReward = reward * positionSize;
  const winRateBreakeven = risk + reward > 0 ? ((risk / (risk + reward)) * 100).toFixed(1) : '50.0';

  return (
    <Card>
      <h3 className="text-sm font-semibold mb-4" style={{ color: 'var(--text-secondary)' }}>
        Risk/Reward Calculator
      </h3>
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-3">
          {[
            { label: 'Entry Price', value: entryPrice, setter: setEntryPrice },
            { label: 'Stop Price', value: stopPrice, setter: setStopPrice },
            { label: 'Target Price', value: targetPrice, setter: setTargetPrice },
            { label: 'Position Size', value: positionSize, setter: setPositionSize },
          ].map(({ label, value, setter }) => (
            <div key={label} className="flex items-center gap-2">
              <label className="text-xs w-24" style={{ color: 'var(--text-muted)' }}>{label}</label>
              <input
                type="number"
                value={value}
                onChange={(e) => setter(Number(e.target.value))}
                className="flex-1 px-2 py-1.5 rounded-lg text-sm font-mono text-right"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                step={label === 'Position Size' ? 1 : 0.01}
              />
            </div>
          ))}
        </div>
        <div className="space-y-2">
          <div className="rounded-xl p-3 text-center" style={{ background: 'var(--bg-input)' }}>
            <span className="text-2xl font-bold font-mono" style={{ color: Number(rr) >= 2 ? 'var(--green)' : Number(rr) >= 1 ? 'var(--orange)' : 'var(--red)' }}>
              {rr}R
            </span>
            <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Risk/Reward Ratio</p>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div className="rounded-lg p-2 text-center" style={{ background: 'var(--bg-input)' }}>
              <span className="text-sm font-bold font-mono" style={{ color: 'var(--red)' }}>${dollarRisk.toFixed(0)}</span>
              <p className="text-[9px]" style={{ color: 'var(--text-muted)' }}>$ Risk</p>
            </div>
            <div className="rounded-lg p-2 text-center" style={{ background: 'var(--bg-input)' }}>
              <span className="text-sm font-bold font-mono" style={{ color: 'var(--green)' }}>${dollarReward.toFixed(0)}</span>
              <p className="text-[9px]" style={{ color: 'var(--text-muted)' }}>$ Reward</p>
            </div>
          </div>
          <div className="rounded-lg p-2 text-center" style={{ background: 'var(--bg-input)' }}>
            <span className="text-xs font-mono" style={{ color: 'var(--accent)' }}>Breakeven Win Rate: {winRateBreakeven}%</span>
          </div>
        </div>
      </div>
    </Card>
  );
}

/* ─────────────────────────── Stop Type Comparison Matrix ─────────────────────────── */

function StopComparisonMatrix() {
  const comparisons = [
    { type: 'ATR', adapts: true, structure: false, trails: false, best: 'Volatile / trend-following', riskStyle: 'Dynamic' },
    { type: 'Fixed $', adapts: false, structure: false, trails: false, best: 'Stable instruments / scalps', riskStyle: 'Static' },
    { type: 'Percentage', adapts: false, structure: false, trails: false, best: 'Multi-price instruments', riskStyle: 'Proportional' },
    { type: 'Swing', adapts: false, structure: true, trails: false, best: 'Structure-based entries', riskStyle: 'Structural' },
    { type: 'ATR Trail', adapts: true, structure: false, trails: true, best: 'Trend riders', riskStyle: 'Dynamic + Trail' },
    { type: 'Breakeven', adapts: true, structure: false, trails: false, best: 'Capital preservation', riskStyle: 'Dynamic + BE' },
  ];

  return (
    <Card>
      <h3 className="text-sm font-semibold mb-4" style={{ color: 'var(--text-secondary)' }}>
        Stop Type Comparison
      </h3>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              <th className="text-left py-2 pr-3" style={{ color: 'var(--text-muted)' }}>Type</th>
              <th className="text-center py-2 px-2" style={{ color: 'var(--text-muted)' }}>Volatility Aware</th>
              <th className="text-center py-2 px-2" style={{ color: 'var(--text-muted)' }}>Structure</th>
              <th className="text-center py-2 px-2" style={{ color: 'var(--text-muted)' }}>Trails</th>
              <th className="text-left py-2 pl-3" style={{ color: 'var(--text-muted)' }}>Best For</th>
            </tr>
          </thead>
          <tbody>
            {comparisons.map((c) => (
              <tr key={c.type} style={{ borderBottom: '1px solid var(--border)' }}>
                <td className="py-2 pr-3 font-medium" style={{ color: 'var(--text-primary)' }}>{c.type}</td>
                {[c.adapts, c.structure, c.trails].map((val, i) => (
                  <td key={i} className="text-center py-2 px-2">
                    {val ? (
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--green)" strokeWidth="2.5" className="inline">
                        <polyline points="20 6 9 17 4 12" />
                      </svg>
                    ) : (
                      <span style={{ color: 'var(--text-muted)' }}>--</span>
                    )}
                  </td>
                ))}
                <td className="py-2 pl-3" style={{ color: 'var(--text-muted)' }}>{c.best}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

/* ─────────────────────────── Position Sizing Visualizer ─────────────────────────── */

function PositionSizingVisualizer() {
  const [accountSize, setAccountSize] = useState(25000);
  const [riskPct, setRiskPct] = useState(1.0);
  const [stopDistance, setStopDistance] = useState(1.5);

  const dollarRisk = accountSize * (riskPct / 100);
  const shares = stopDistance > 0 ? Math.floor(dollarRisk / stopDistance) : 0;
  const actualRisk = shares * stopDistance;
  const actualRiskPct = ((actualRisk / accountSize) * 100).toFixed(2);

  // Visual bar segments
  const barSegments = Math.min(shares, 50);
  const riskBarWidth = Math.min((riskPct / 5) * 100, 100);

  return (
    <Card>
      <h3 className="text-sm font-semibold mb-4" style={{ color: 'var(--text-secondary)' }}>
        Position Sizing
      </h3>
      <div className="grid grid-cols-3 gap-4 mb-4">
        {[
          { label: 'Account Size', value: accountSize, setter: setAccountSize, prefix: '$', step: 1000 },
          { label: 'Risk %', value: riskPct, setter: setRiskPct, prefix: '', step: 0.25 },
          { label: 'Stop Distance ($)', value: stopDistance, setter: setStopDistance, prefix: '$', step: 0.1 },
        ].map(({ label, value, setter, step }) => (
          <div key={label}>
            <label className="text-[10px] mb-1 block" style={{ color: 'var(--text-muted)' }}>{label}</label>
            <input
              type="number"
              value={value}
              onChange={(e) => setter(Number(e.target.value))}
              className="w-full px-2 py-1.5 rounded-lg text-sm font-mono text-right"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
              step={step}
            />
          </div>
        ))}
      </div>

      {/* Risk visualization */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-1">
          <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Account Risk</span>
          <span className="text-[10px] font-mono" style={{ color: riskPct > 2 ? 'var(--red)' : 'var(--green)' }}>{actualRiskPct}%</span>
        </div>
        <div className="h-3 rounded-full overflow-hidden" style={{ background: 'var(--bg-input)' }}>
          <div
            className="h-full rounded-full transition-all"
            style={{
              width: `${riskBarWidth}%`,
              background: riskPct > 2 ? 'var(--red)' : riskPct > 1 ? 'var(--orange)' : 'var(--green)',
            }}
          />
        </div>
        <div className="flex justify-between mt-1">
          <span className="text-[9px]" style={{ color: 'var(--text-muted)' }}>0%</span>
          <span className="text-[9px]" style={{ color: 'var(--text-muted)' }}>Conservative (1%)</span>
          <span className="text-[9px]" style={{ color: 'var(--text-muted)' }}>Aggressive (2%+)</span>
        </div>
      </div>

      {/* Results */}
      <div className="grid grid-cols-3 gap-2">
        <div className="rounded-lg p-3 text-center" style={{ background: 'var(--bg-input)' }}>
          <span className="text-xl font-bold font-mono" style={{ color: 'var(--accent)' }}>{shares}</span>
          <p className="text-[9px]" style={{ color: 'var(--text-muted)' }}>Shares</p>
        </div>
        <div className="rounded-lg p-3 text-center" style={{ background: 'var(--bg-input)' }}>
          <span className="text-xl font-bold font-mono" style={{ color: 'var(--red)' }}>${actualRisk.toFixed(0)}</span>
          <p className="text-[9px]" style={{ color: 'var(--text-muted)' }}>Max Loss</p>
        </div>
        <div className="rounded-lg p-3 text-center" style={{ background: 'var(--bg-input)' }}>
          <span className="text-xl font-bold font-mono" style={{ color: 'var(--green)' }}>${(actualRisk * 2).toFixed(0)}</span>
          <p className="text-[9px]" style={{ color: 'var(--text-muted)' }}>2R Target</p>
        </div>
      </div>
    </Card>
  );
}

/* ─────────────────────────── Pack Detail Modal ─────────────────────────── */

function PackDetailModal({ pack, onClose, onUpdate }: { pack: RiskPack; onClose: () => void; onUpdate: (id: string, params: PackParam[]) => void }) {
  const [localParams, setLocalParams] = useState<PackParam[]>(pack.params);
  const [detailTab, setDetailTab] = useState<'visual' | 'params' | 'help'>('visual');

  const updateParam = (key: string, val: number) => {
    setLocalParams((prev) => prev.map((p) => (p.key === key ? { ...p, value: val } : p)));
  };

  const updatedPack = { ...pack, params: localParams };

  return (
    <Modal title="" isOpen onClose={onClose} width="800px">
      <div className="flex items-start gap-4 -mt-2 mb-6">
        <div className="w-20 h-20 rounded-2xl flex items-center justify-center flex-shrink-0" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
          <StopTypeIcon stopType={pack.stopType} size={56} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <h2 className="text-xl font-bold">{pack.name}</h2>
            <span className="text-xs px-2 py-0.5 rounded-full" style={{ color: categoryColors[pack.category]?.color, background: categoryColors[pack.category]?.bg }}>
              {pack.category}
            </span>
          </div>
          <div className="flex items-center gap-3 text-xs" style={{ color: 'var(--text-muted)' }}>
            <span>Stop: <strong style={{ color: 'var(--red)' }}>{pack.stopSummary}</strong></span>
            <span>Target: <strong style={{ color: 'var(--green)' }}>{pack.targetSummary}</strong></span>
          </div>
        </div>
      </div>

      <div className="flex gap-1 mb-5 p-1 rounded-xl" style={{ background: 'var(--bg-input)' }}>
        {(['visual', 'params', 'help'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setDetailTab(tab)}
            className="flex-1 py-2 text-xs font-medium rounded-lg transition-all"
            style={{
              background: detailTab === tab ? 'var(--bg-card)' : 'transparent',
              color: detailTab === tab ? 'var(--text-primary)' : 'var(--text-muted)',
              boxShadow: detailTab === tab ? '0 1px 3px rgba(0,0,0,0.2)' : 'none',
            }}
          >
            {tab === 'visual' ? 'Stop Placement' : tab === 'params' ? 'Parameters' : 'How It Works'}
          </button>
        ))}
      </div>

      {detailTab === 'visual' && (
        <div className="space-y-4">
          <div className="rounded-xl p-4 flex items-center justify-center" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
            <StopPlacementDiagram pack={updatedPack} size={700} height={200} />
          </div>
          {/* Key metrics */}
          <div className="grid grid-cols-3 gap-3">
            {(() => {
              const stopParam = localParams.find((p) => p.key.includes('stop'));
              const targetParam = localParams.find((p) => p.key.includes('target'));
              const stopVal = stopParam ? Number(stopParam.value) : 0;
              const targetVal = targetParam ? Number(targetParam.value) : 0;
              const rr = stopVal > 0 && targetVal > 0 ? (targetVal / stopVal).toFixed(1) : '---';
              return (
                <>
                  <MetricCard label="Stop Distance" value={pack.stopSummary} delta={pack.stopType} positive />
                  <MetricCard label="Target" value={pack.targetSummary} delta={targetVal === 0 ? 'No fixed target' : 'Fixed'} positive={targetVal > 0} />
                  <MetricCard label="Risk:Reward" value={`${rr}`} delta={Number(rr) >= 2 ? 'Favorable' : Number(rr) >= 1 ? 'Moderate' : 'Low'} positive={Number(rr) >= 1.5} />
                </>
              );
            })()}
          </div>
        </div>
      )}

      {detailTab === 'params' && (
        <div className="space-y-4">
          {localParams.map((param) => (
            <div
              key={param.key}
              className="px-4 py-3 rounded-xl"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>{param.label}</span>
                <span className="text-sm font-mono font-bold" style={{ color: 'var(--accent)' }}>{String(param.value)}</span>
              </div>
              {param.help && (
                <p className="text-[10px] mb-2" style={{ color: 'var(--text-muted)' }}>{param.help}</p>
              )}
              <div className="relative h-6 flex items-center">
                <div className="absolute inset-x-0 h-1.5 rounded-full" style={{ background: 'var(--bg-primary)' }} />
                {(() => {
                  const min = param.min ?? 0;
                  const max = param.max ?? 100;
                  const pct = ((Number(param.value) - min) / (max - min)) * 100;
                  return (
                    <>
                      <div className="absolute left-0 h-1.5 rounded-full" style={{ width: `${pct}%`, background: 'var(--accent)', transition: 'width 0.15s ease' }} />
                      <input
                        type="range"
                        min={min}
                        max={max}
                        step={param.step || (param.type === 'float' ? 0.1 : 1)}
                        value={Number(param.value)}
                        onChange={(e) => updateParam(param.key, Number(e.target.value))}
                        className="absolute inset-x-0 w-full h-full opacity-0 cursor-pointer"
                        style={{ zIndex: 2 }}
                      />
                      <div
                        className="absolute w-4 h-4 rounded-full border-2 shadow-lg pointer-events-none"
                        style={{ left: `calc(${pct}% - 8px)`, background: 'var(--bg-card)', borderColor: 'var(--accent)', transition: 'left 0.15s ease' }}
                      />
                    </>
                  );
                })()}
              </div>
              <div className="flex justify-between mt-1">
                <span className="text-[9px] font-mono" style={{ color: 'var(--text-muted)' }}>{param.min}</span>
                <span className="text-[9px] font-mono" style={{ color: 'var(--text-muted)' }}>{param.max}</span>
              </div>
            </div>
          ))}
          {/* Live diagram updates */}
          <div className="rounded-xl p-3 flex items-center justify-center" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
            <StopPlacementDiagram pack={updatedPack} size={600} height={120} />
          </div>
          <div className="flex gap-3 pt-1">
            <button
              className="px-4 py-2 rounded-lg text-sm font-medium flex-1"
              style={{ background: 'var(--accent)', color: 'white' }}
              onClick={() => { onUpdate(pack.id, localParams); onClose(); }}
            >
              Save Parameters
            </button>
            <button
              className="px-4 py-2 rounded-lg text-sm"
              style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
              onClick={() => setLocalParams(pack.params)}
            >
              Reset
            </button>
          </div>
        </div>
      )}

      {detailTab === 'help' && (
        <div className="space-y-3">
          {pack.contextualHelp.map((tip, i) => (
            <div
              key={i}
              className="flex gap-3 px-4 py-3 rounded-xl"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
            >
              <div className="w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 text-xs font-bold" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                {i + 1}
              </div>
              <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>{tip}</p>
            </div>
          ))}
          {/* Stop lifecycle diagram for trailing/BE types */}
          {(pack.stopType === 'Trail' || pack.stopType === 'BE') && (
            <div className="rounded-xl p-4" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
              <h4 className="text-xs font-semibold mb-3" style={{ color: 'var(--text-secondary)' }}>Stop Lifecycle</h4>
              <div className="flex items-center gap-2">
                {(pack.stopType === 'Trail'
                  ? ['Entry', 'Initial ATR Stop', 'Profit Threshold', 'Trail Activates', 'Trail Ratchets']
                  : ['Entry', 'Initial ATR Stop', 'Profit Threshold', 'Stop Moves to BE', 'Trade Exits']
                ).map((step, i, arr) => (
                  <div key={i} className="flex items-center gap-2">
                    <div className="flex flex-col items-center">
                      <div
                        className="w-7 h-7 rounded-full flex items-center justify-center text-[10px] font-bold"
                        style={{
                          background: i === arr.length - 1 ? 'var(--accent)' : 'var(--bg-card)',
                          color: i === arr.length - 1 ? 'white' : 'var(--text-secondary)',
                          border: '1px solid var(--border)',
                        }}
                      >
                        {i + 1}
                      </div>
                      <span className="text-[8px] mt-1 text-center w-16" style={{ color: 'var(--text-muted)' }}>{step}</span>
                    </div>
                    {i < arr.length - 1 && (
                      <svg width="16" height="8" viewBox="0 0 16 8" fill="none" className="flex-shrink-0">
                        <path d="M0,4 L12,4 M10,1 L14,4 L10,7" stroke="var(--text-muted)" strokeWidth="1" />
                      </svg>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </Modal>
  );
}

/* ═══════════════════════════ Main Component ═══════════════════════════ */

export default function RiskManagementV4() {
  const [packs, setPacks] = useState<RiskPack[]>(initialPacks);
  const [detailPack, setDetailPack] = useState<RiskPack | null>(null);
  const [showNewModal, setShowNewModal] = useState(false);
  const [filterCategory, setFilterCategory] = useState<string>('All');

  const togglePack = useCallback((id: string) => {
    setPacks((prev) => prev.map((p) => (p.id === id ? { ...p, enabled: !p.enabled } : p)));
  }, []);

  const updatePackParams = useCallback((id: string, params: PackParam[]) => {
    setPacks((prev) => prev.map((p) => (p.id === id ? { ...p, params } : p)));
  }, []);

  const enabledCount = packs.filter((p) => p.enabled).length;
  const categories = ['All', 'Volatility', 'Fixed', 'Structure', 'Trailing'];
  const filtered = filterCategory === 'All' ? packs : packs.filter((p) => p.category === filterCategory);

  const grouped = useMemo(() => {
    const map = new Map<string, RiskPack[]>();
    filtered.forEach((p) => {
      if (!map.has(p.category)) map.set(p.category, []);
      map.get(p.category)!.push(p);
    });
    return map;
  }, [filtered]);

  return (
    <div>
      <style>{`
        @keyframes pulse-ring {
          0% { transform: scale(1); opacity: 0.6; }
          100% { transform: scale(2.5); opacity: 0; }
        }
        @keyframes fade-in-up {
          0% { opacity: 0; transform: translateY(8px); }
          100% { opacity: 1; transform: translateY(0); }
        }
        .risk-card { animation: fade-in-up 0.4s ease forwards; }
        .risk-card:nth-child(1) { animation-delay: 0.0s; }
        .risk-card:nth-child(2) { animation-delay: 0.05s; }
        .risk-card:nth-child(3) { animation-delay: 0.1s; }
        .risk-card:nth-child(4) { animation-delay: 0.15s; }
        .risk-card:nth-child(5) { animation-delay: 0.2s; }
        .risk-card:nth-child(6) { animation-delay: 0.25s; }
        .risk-card:nth-child(7) { animation-delay: 0.3s; }
      `}</style>

      <PageHeader
        title="Risk Management Packs"
        subtitle="Stop placement, position sizing, and risk/reward -- visual tools to manage every trade"
        actions={
          <button
            onClick={() => setShowNewModal(true)}
            className="px-4 py-2 rounded-lg text-sm font-medium"
            style={{ background: 'var(--accent)', color: 'white' }}
          >
            + New Pack
          </button>
        }
      />

      {/* Summary Dashboard */}
      <div className="grid grid-cols-4 gap-3 mb-6">
        <MetricCard label="Total Packs" value={String(packs.length)} delta={`${enabledCount} enabled`} positive />
        <MetricCard label="Stop Types" value={String(new Set(packs.map((p) => p.stopType)).size)} delta="Unique types" positive />
        <MetricCard label="Trailing Stops" value={String(packs.filter((p) => p.category === 'Trailing').length)} delta="Available" positive />
        <MetricCard label="Active Pack" value={packs.find((p) => p.enabled)?.name || 'None'} delta={packs.find((p) => p.enabled)?.stopSummary || '-'} positive={enabledCount > 0} />
      </div>

      {/* Interactive tools */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
        <RiskRewardCalculator />
        <PositionSizingVisualizer />
      </div>

      {/* Stop comparison matrix */}
      <div className="mb-6">
        <StopComparisonMatrix />
      </div>

      {/* Category filter */}
      <div className="flex items-center gap-2 mb-5">
        {categories.map((cat) => (
          <button
            key={cat}
            onClick={() => setFilterCategory(cat)}
            className="px-4 py-2 rounded-lg text-xs font-medium transition-all"
            style={{
              background: filterCategory === cat ? (cat === 'All' ? 'var(--accent)' : categoryColors[cat]?.color || 'var(--accent)') : 'var(--bg-input)',
              color: filterCategory === cat ? 'white' : 'var(--text-secondary)',
              border: filterCategory === cat ? 'none' : '1px solid var(--border)',
            }}
          >
            {cat}
          </button>
        ))}
      </div>

      {/* Pack Cards */}
      <div className="space-y-8">
        {Array.from(grouped.entries()).map(([groupName, groupPacks]) => (
          <div key={groupName}>
            <div className="flex items-center gap-3 mb-3">
              <div className="w-1.5 h-5 rounded-full" style={{ background: categoryColors[groupName]?.color || 'var(--accent)' }} />
              <h3 className="text-sm font-semibold" style={{ color: 'var(--text-secondary)' }}>{groupName}</h3>
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{groupPacks.length} pack{groupPacks.length !== 1 ? 's' : ''}</span>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {groupPacks.map((pack) => (
                <div key={pack.id} className="risk-card">
                  <Card className={`relative overflow-hidden ${!pack.enabled ? 'opacity-60' : ''}`}>
                    <div
                      className="absolute top-0 left-0 right-0 h-0.5"
                      style={{
                        background: pack.enabled
                          ? `linear-gradient(90deg, ${categoryColors[pack.category]?.color || 'var(--accent)'}, transparent)`
                          : 'var(--border)',
                      }}
                    />

                    <div className="flex gap-4">
                      {/* Stop type icon */}
                      <div
                        className="w-16 h-16 rounded-xl flex items-center justify-center flex-shrink-0 cursor-pointer transition-transform hover:scale-105"
                        style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
                        onClick={() => setDetailPack(pack)}
                      >
                        <StopTypeIcon stopType={pack.stopType} />
                      </div>

                      <div className="flex-1 min-w-0">
                        <div className="flex items-start justify-between mb-1.5">
                          <div>
                            <div className="flex items-center gap-2 mb-0.5">
                              <button
                                onClick={() => setDetailPack(pack)}
                                className="font-semibold text-sm hover:underline"
                                style={{ color: 'var(--text-primary)' }}
                              >
                                {pack.name}
                              </button>
                              <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ background: categoryColors[pack.category]?.bg, color: categoryColors[pack.category]?.color }}>
                                {pack.stopType}
                              </span>
                            </div>
                            <div className="flex items-center gap-3 text-[11px]" style={{ color: 'var(--text-muted)' }}>
                              <span>Stop: <strong style={{ color: 'var(--red)' }}>{pack.stopSummary}</strong></span>
                              <span>Target: <strong style={{ color: 'var(--green)' }}>{pack.targetSummary}</strong></span>
                            </div>
                          </div>

                          <button
                            onClick={() => togglePack(pack.id)}
                            className="w-10 h-6 rounded-full relative flex-shrink-0 transition-colors"
                            style={{
                              background: pack.enabled ? 'var(--accent)' : 'var(--bg-input)',
                              border: pack.enabled ? 'none' : '1px solid var(--border)',
                            }}
                          >
                            <div
                              className="w-4 h-4 rounded-full absolute top-1 transition-all"
                              style={{
                                background: pack.enabled ? 'white' : 'var(--text-muted)',
                                left: pack.enabled ? '22px' : '4px',
                              }}
                            />
                          </button>
                        </div>

                        {/* Mini stop placement diagram */}
                        <div className="mt-2 rounded-lg overflow-hidden" style={{ background: 'var(--bg-input)' }}>
                          <StopPlacementDiagram pack={pack} size={320} height={60} />
                        </div>

                        {/* Params preview */}
                        <div className="flex flex-wrap gap-1.5 mt-2">
                          {pack.params.map((p) => (
                            <span key={p.key} className="text-[10px] font-mono px-1.5 py-0.5 rounded" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>
                              {p.label}: {String(p.value)}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  </Card>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Detail Modal */}
      {detailPack && (
        <PackDetailModal
          pack={detailPack}
          onClose={() => setDetailPack(null)}
          onUpdate={updatePackParams}
        />
      )}

      {/* New Pack Modal */}
      <Modal title="Create Risk Pack" isOpen={showNewModal} onClose={() => setShowNewModal(false)} width="520px">
        <NewPackForm onClose={() => setShowNewModal(false)} />
      </Modal>
    </div>
  );
}

/* ─────────────────────────── New Pack Form ─────────────────────────── */

function NewPackForm({ onClose }: { onClose: () => void }) {
  const [template, setTemplate] = useState('ATR-Based');
  const [versionName, setVersionName] = useState('');

  const templates = [
    { name: 'ATR-Based', category: 'Volatility', desc: 'Volatility-adaptive stops using ATR multiplier' },
    { name: 'Fixed Dollar', category: 'Fixed', desc: 'Constant dollar distance from entry' },
    { name: 'Percentage', category: 'Fixed', desc: 'Percentage-based stop distance' },
    { name: 'Swing Stop', category: 'Structure', desc: 'Stop at recent swing high/low' },
    { name: 'ATR Trailing', category: 'Trailing', desc: 'ATR stop that trails in your favor' },
    { name: 'Breakeven', category: 'Trailing', desc: 'Stop jumps to breakeven after profit threshold' },
  ];

  const selected = templates.find((t) => t.name === template);

  return (
    <div className="space-y-5">
      <div>
        <label className="text-xs font-medium mb-2 block" style={{ color: 'var(--text-secondary)' }}>Stop Type Template</label>
        <div className="grid grid-cols-1 gap-2">
          {templates.map((t) => {
            const isSelected = template === t.name;
            return (
              <button
                key={t.name}
                onClick={() => setTemplate(t.name)}
                className="flex items-center gap-3 px-4 py-3 rounded-xl text-left transition-all"
                style={{
                  background: isSelected ? `${categoryColors[t.category]?.color}10` : 'var(--bg-input)',
                  border: isSelected ? `1px solid ${categoryColors[t.category]?.color}60` : '1px solid var(--border)',
                }}
              >
                <div className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0" style={{ background: isSelected ? categoryColors[t.category]?.bg : 'var(--bg-primary)' }}>
                  <StopTypeIcon stopType={t.name.includes('ATR') ? (t.name.includes('Trail') ? 'Trail' : 'ATR') : t.name === 'Swing Stop' ? 'Swing' : t.name === 'Breakeven' ? 'BE' : t.name === 'Percentage' ? 'Pct' : 'Fixed $'} size={32} />
                </div>
                <div>
                  <span className="text-sm font-medium" style={{ color: isSelected ? categoryColors[t.category]?.color : 'var(--text-primary)' }}>{t.name}</span>
                  <p className="text-[10px] mt-0.5" style={{ color: 'var(--text-muted)' }}>{t.desc}</p>
                </div>
                {isSelected && (
                  <svg className="ml-auto" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={categoryColors[t.category]?.color} strokeWidth="2">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                )}
              </button>
            );
          })}
        </div>
      </div>

      <div>
        <label className="text-xs font-medium mb-1 block" style={{ color: 'var(--text-secondary)' }}>Version Name</label>
        <input
          className="w-full px-3 py-2 rounded-lg text-sm"
          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
          placeholder="e.g. Tight, Wide, Conservative..."
          value={versionName}
          onChange={(e) => setVersionName(e.target.value)}
        />
      </div>

      <div className="flex gap-3 pt-1">
        <button className="px-4 py-2.5 rounded-lg text-sm font-medium flex-1" style={{ background: 'var(--accent)', color: 'white' }} onClick={onClose}>
          Create Pack
        </button>
        <button className="px-4 py-2.5 rounded-lg text-sm flex-1" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }} onClick={onClose}>
          Cancel
        </button>
      </div>
    </div>
  );
}
