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
  value: number;
}

interface PackTrigger {
  name: string;
  direction: 'LONG' | 'SHORT' | 'BOTH';
  exec: string;
}

interface Pack {
  id: string;
  name: string;
  version: string;
  category: string;
  enabled: boolean;
  isDefault: boolean;
  currentState: string;
  currentStateLabel: string;
  strategiesUsing: number;
  lastTriggered: string;
  stateDistribution: Record<string, number>;
  params: PackParam[];
  outputs: string[];
  triggers: PackTrigger[];
}

/* ─────────────────────────── Mock Data ─────────────────────────── */

const initialPacks: Pack[] = [
  {
    id: 'ema-stack-default',
    name: 'EMA Stack',
    version: 'Default',
    category: 'Moving Averages',
    enabled: true,
    isDefault: true,
    currentState: 'SML',
    currentStateLabel: 'Full Bull Stack',
    strategiesUsing: 4,
    lastTriggered: '2h ago',
    stateDistribution: { SML: 42, SLM: 8, MSL: 15, MLS: 12, LSM: 10, LMS: 13 },
    params: [
      { key: 'short_period', label: 'Short', value: 9 },
      { key: 'mid_period', label: 'Mid', value: 21 },
      { key: 'long_period', label: 'Long', value: 200 },
    ],
    outputs: ['SML', 'SLM', 'MSL', 'MLS', 'LSM', 'LMS'],
    triggers: [
      { name: 'Short > Mid Cross', direction: 'LONG', exec: '[C]' },
      { name: 'Short < Mid Cross', direction: 'SHORT', exec: '[C]' },
      { name: 'Mid > Long Cross', direction: 'LONG', exec: '[C]' },
      { name: 'Mid < Long Cross', direction: 'SHORT', exec: '[C]' },
    ],
  },
  {
    id: 'macd-line-default',
    name: 'MACD Line',
    version: 'Default',
    category: 'Momentum',
    enabled: true,
    isDefault: true,
    currentState: 'M>S+',
    currentStateLabel: 'Strong Bull',
    strategiesUsing: 3,
    lastTriggered: '45m ago',
    stateDistribution: { 'M>S+': 35, 'M>S-': 20, 'M<S-': 25, 'M<S+': 20 },
    params: [
      { key: 'fast_period', label: 'Fast', value: 12 },
      { key: 'slow_period', label: 'Slow', value: 26 },
      { key: 'signal_period', label: 'Signal', value: 9 },
    ],
    outputs: ['M>S+', 'M>S-', 'M<S-', 'M<S+'],
    triggers: [
      { name: 'Bullish Cross', direction: 'LONG', exec: '[C]' },
      { name: 'Bearish Cross', direction: 'SHORT', exec: '[C]' },
      { name: 'Zero Line Cross Up', direction: 'LONG', exec: '[C]' },
      { name: 'Zero Line Cross Down', direction: 'SHORT', exec: '[C]' },
    ],
  },
  {
    id: 'macd-hist-default',
    name: 'MACD Histogram',
    version: 'Default',
    category: 'Momentum',
    enabled: false,
    isDefault: true,
    currentState: 'H+dn',
    currentStateLabel: 'Decelerating Bull',
    strategiesUsing: 0,
    lastTriggered: 'never',
    stateDistribution: { 'H+up': 28, 'H+dn': 22, 'H-dn': 30, 'H-up': 20 },
    params: [
      { key: 'fast_period', label: 'Fast', value: 12 },
      { key: 'slow_period', label: 'Slow', value: 26 },
      { key: 'signal_period', label: 'Signal', value: 9 },
    ],
    outputs: ['H+up', 'H+dn', 'H-dn', 'H-up'],
    triggers: [
      { name: 'Histogram Flip Bullish', direction: 'LONG', exec: '[C]' },
      { name: 'Histogram Flip Bearish', direction: 'SHORT', exec: '[C]' },
    ],
  },
  {
    id: 'vwap-default',
    name: 'VWAP',
    version: 'Default',
    category: 'Volume',
    enabled: true,
    isDefault: true,
    currentState: '>V',
    currentStateLabel: 'Above VWAP Zone',
    strategiesUsing: 5,
    lastTriggered: '12m ago',
    stateDistribution: { '>+2\u03c3': 5, '>+1\u03c3': 12, '>V': 28, '@V': 15, '<V': 22, '<-1\u03c3': 10, '<-2\u03c3': 8 },
    params: [
      { key: 'sd1_mult', label: 'SD1', value: 1.0 },
      { key: 'sd2_mult', label: 'SD2', value: 2.0 },
    ],
    outputs: ['>+2\u03c3', '>+1\u03c3', '>V', '@V', '<V', '<-1\u03c3', '<-2\u03c3'],
    triggers: [
      { name: 'Cross Above VWAP', direction: 'LONG', exec: '[C]' },
      { name: 'Cross Below VWAP', direction: 'SHORT', exec: '[C]' },
      { name: 'Enter Upper Extreme', direction: 'SHORT', exec: '[C]' },
      { name: 'Enter Lower Extreme', direction: 'LONG', exec: '[C]' },
      { name: 'Return to VWAP Zone', direction: 'BOTH', exec: '[C]' },
    ],
  },
  {
    id: 'rvol-default',
    name: 'Relative Volume',
    version: 'Default',
    category: 'Volume',
    enabled: false,
    isDefault: true,
    currentState: 'NORMAL',
    currentStateLabel: 'Normal Volume',
    strategiesUsing: 1,
    lastTriggered: '3h ago',
    stateDistribution: { EXTREME: 3, HIGH: 12, NORMAL: 55, LOW: 20, MINIMAL: 10 },
    params: [
      { key: 'sma_period', label: 'SMA Period', value: 20 },
      { key: 'high_threshold', label: 'High', value: 1.5 },
      { key: 'extreme_threshold', label: 'Extreme', value: 3.0 },
    ],
    outputs: ['EXTREME', 'HIGH', 'NORMAL', 'LOW', 'MINIMAL'],
    triggers: [
      { name: 'Volume Spike', direction: 'BOTH', exec: '[C]' },
      { name: 'Extreme Volume', direction: 'BOTH', exec: '[C]' },
      { name: 'Volume Fade', direction: 'BOTH', exec: '[C]' },
    ],
  },
  {
    id: 'ema-stack-scalping',
    name: 'EMA Stack',
    version: 'Scalping',
    category: 'Moving Averages',
    enabled: true,
    isDefault: false,
    currentState: 'MSL',
    currentStateLabel: 'Mid Leading',
    strategiesUsing: 2,
    lastTriggered: '30m ago',
    stateDistribution: { SML: 30, SLM: 10, MSL: 25, MLS: 15, LSM: 8, LMS: 12 },
    params: [
      { key: 'short_period', label: 'Short', value: 5 },
      { key: 'mid_period', label: 'Mid', value: 13 },
      { key: 'long_period', label: 'Long', value: 50 },
    ],
    outputs: ['SML', 'SLM', 'MSL', 'MLS', 'LSM', 'LMS'],
    triggers: [
      { name: 'Short > Mid Cross', direction: 'LONG', exec: '[C]' },
      { name: 'Short < Mid Cross', direction: 'SHORT', exec: '[C]' },
      { name: 'Mid > Long Cross', direction: 'LONG', exec: '[C]' },
      { name: 'Mid < Long Cross', direction: 'SHORT', exec: '[C]' },
    ],
  },
];

/* ─────────────────────────── Inline SVG Indicator Icons ─────────────────────────── */

function EmaStackIcon({ state, size = 64 }: { state: string; size?: number }) {
  // Three moving average lines with different slopes depending on state
  const stateLines: Record<string, { s: number[]; m: number[]; l: number[] }> = {
    SML: { s: [38, 30, 18, 10], m: [42, 36, 28, 22], l: [50, 48, 44, 40] },
    MSL: { s: [42, 36, 28, 22], m: [38, 30, 18, 10], l: [50, 48, 44, 40] },
    LSM: { s: [50, 48, 44, 40], m: [42, 36, 28, 22], l: [38, 30, 18, 10] },
    LMS: { s: [50, 48, 44, 40], m: [38, 30, 18, 10], l: [42, 36, 28, 22] },
    SLM: { s: [38, 30, 18, 10], m: [50, 48, 44, 40], l: [42, 36, 28, 22] },
    MLS: { s: [42, 36, 28, 22], m: [50, 48, 44, 40], l: [38, 30, 18, 10] },
  };
  const lines = stateLines[state] || stateLines.SML;
  const xs = [8, 24, 42, 56];
  const mkPath = (pts: number[]) => pts.map((y, i) => `${i === 0 ? 'M' : 'L'}${xs[i]},${y}`).join(' ');

  return (
    <svg width={size} height={size} viewBox="0 0 64 64" fill="none">
      <path d={mkPath(lines.l)} stroke="var(--blue)" strokeWidth="2" strokeLinecap="round" opacity="0.5" />
      <path d={mkPath(lines.m)} stroke="var(--orange)" strokeWidth="2" strokeLinecap="round" opacity="0.7" />
      <path d={mkPath(lines.s)} stroke="var(--green)" strokeWidth="2.5" strokeLinecap="round" />
      {/* Tiny glow on the fast line */}
      <path d={mkPath(lines.s)} stroke="var(--green)" strokeWidth="6" strokeLinecap="round" opacity="0.15" />
    </svg>
  );
}

function MacdLineIcon({ size = 64 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" fill="none">
      {/* Zero line */}
      <line x1="4" y1="32" x2="60" y2="32" stroke="var(--text-muted)" strokeWidth="0.5" strokeDasharray="2 2" opacity="0.5" />
      {/* MACD line */}
      <path d="M6,40 Q16,44 24,36 Q32,26 40,22 Q48,18 58,24" stroke="var(--blue)" strokeWidth="2" strokeLinecap="round" />
      {/* Signal line */}
      <path d="M6,38 Q16,42 24,40 Q32,34 40,28 Q48,22 58,20" stroke="var(--orange)" strokeWidth="1.5" strokeLinecap="round" strokeDasharray="3 2" />
      {/* Cross point glow */}
      <circle cx="20" cy="40" r="3" fill="var(--green)" opacity="0.4" />
      <circle cx="20" cy="40" r="1.5" fill="var(--green)" />
    </svg>
  );
}

function MacdHistIcon({ size = 64 }: { size?: number }) {
  const bars = [
    { x: 6, h: 8, positive: true, fading: false },
    { x: 14, h: 14, positive: true, fading: false },
    { x: 22, h: 10, positive: true, fading: true },
    { x: 30, h: 4, positive: true, fading: true },
    { x: 38, h: -6, positive: false, fading: false },
    { x: 46, h: -12, positive: false, fading: false },
    { x: 54, h: -8, positive: false, fading: true },
  ];
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" fill="none">
      <line x1="4" y1="32" x2="60" y2="32" stroke="var(--text-muted)" strokeWidth="0.5" opacity="0.5" />
      {bars.map((b, i) => (
        <rect
          key={i}
          x={b.x}
          y={b.h > 0 ? 32 - b.h * 1.5 : 32}
          width="6"
          height={Math.abs(b.h) * 1.5}
          rx="1"
          fill={b.positive ? 'var(--green)' : 'var(--red)'}
          opacity={b.fading ? 0.4 : 0.8}
        />
      ))}
    </svg>
  );
}

function VwapIcon({ size = 64 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" fill="none">
      {/* Upper band */}
      <path d="M4,18 Q16,14 28,20 Q40,16 52,22 L60,20" stroke="var(--accent)" strokeWidth="1" opacity="0.3" />
      {/* VWAP line */}
      <path d="M4,30 Q16,26 28,32 Q40,28 52,34 L60,32" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" />
      {/* Lower band */}
      <path d="M4,42 Q16,38 28,44 Q40,40 52,46 L60,44" stroke="var(--accent)" strokeWidth="1" opacity="0.3" />
      {/* Fill between bands */}
      <path d="M4,18 Q16,14 28,20 Q40,16 52,22 L60,20 L60,44 Q52,46 40,40 Q28,44 16,38 L4,42 Z" fill="var(--accent)" opacity="0.06" />
      {/* Price line crossing above */}
      <path d="M4,36 Q12,34 20,28 Q28,24 36,26 Q44,22 52,18 L60,16" stroke="var(--green)" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function RvolIcon({ size = 64 }: { size?: number }) {
  const bars = [6, 10, 7, 18, 35, 22, 12, 8, 14, 9];
  const maxH = 35;
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" fill="none">
      {bars.map((h, i) => {
        const x = 4 + i * 6;
        const isSpike = h > 25;
        const isHigh = h > 15;
        return (
          <g key={i}>
            <rect
              x={x}
              y={54 - (h / maxH) * 40}
              width="4"
              height={(h / maxH) * 40}
              rx="1"
              fill={isSpike ? 'var(--red)' : isHigh ? 'var(--orange)' : 'var(--accent)'}
              opacity={isSpike ? 0.9 : isHigh ? 0.7 : 0.4}
            />
            {isSpike && (
              <rect
                x={x}
                y={54 - (h / maxH) * 40}
                width="4"
                height={(h / maxH) * 40}
                rx="1"
                fill={isSpike ? 'var(--red)' : 'var(--orange)'}
                opacity={0.2}
                style={{ filter: 'blur(2px)' }}
              />
            )}
          </g>
        );
      })}
      {/* Avg line */}
      <line x1="2" y1="40" x2="62" y2="40" stroke="var(--text-muted)" strokeWidth="0.5" strokeDasharray="2 2" opacity="0.6" />
    </svg>
  );
}

function getPackIcon(pack: Pack, size = 64) {
  if (pack.name === 'EMA Stack') return <EmaStackIcon state={pack.currentState} size={size} />;
  if (pack.name === 'MACD Line') return <MacdLineIcon size={size} />;
  if (pack.name === 'MACD Histogram') return <MacdHistIcon size={size} />;
  if (pack.name === 'VWAP') return <VwapIcon size={size} />;
  if (pack.name === 'Relative Volume') return <RvolIcon size={size} />;
  return <EmaStackIcon state="SML" size={size} />;
}

/* ─────────────────────────── Helpers ─────────────────────────── */

const categoryColors: Record<string, { color: string; bg: string }> = {
  'Moving Averages': { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  Momentum: { color: 'var(--orange)', bg: 'var(--orange-muted)' },
  Volume: { color: 'var(--accent)', bg: 'var(--accent-muted)' },
  Volatility: { color: 'var(--red)', bg: 'var(--red-muted)' },
  Trend: { color: 'var(--green)', bg: 'var(--green-muted)' },
};

const directionColors: Record<string, { color: string; icon: string }> = {
  LONG: { color: 'var(--green)', icon: '\u2191' },
  SHORT: { color: 'var(--red)', icon: '\u2193' },
  BOTH: { color: 'var(--accent)', icon: '\u2195' },
};

function getStateColor(state: string, pack: Pack): string {
  // Bullish states
  if (['SML', 'M>S+', 'H+up', '>V', '>+1\u03c3', '>+2\u03c3', 'EXTREME', 'HIGH'].includes(state)) return 'var(--green)';
  // Bearish states
  if (['LMS', 'M<S-', 'H-dn', '<V', '<-1\u03c3', '<-2\u03c3', 'MINIMAL'].includes(state)) return 'var(--red)';
  // Neutral / transitional
  if (['@V', 'NORMAL'].includes(state)) return 'var(--text-muted)';
  // Mildly bullish/bearish
  if (['SLM', 'MSL', 'M>S-', 'H+dn', 'H-up'].includes(state)) return 'var(--orange)';
  return 'var(--accent)';
}

/* ─────────────────────────── Mini State Distribution Bar ─────────────────────────── */

function StateDistributionBar({ distribution, currentState, pack }: { distribution: Record<string, number>; currentState: string; pack: Pack }) {
  const total = Object.values(distribution).reduce((a, b) => a + b, 0);
  const entries = Object.entries(distribution);

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex rounded-full overflow-hidden h-2" style={{ background: 'var(--bg-input)' }}>
        {entries.map(([state, pct]) => (
          <div
            key={state}
            style={{
              width: `${(pct / total) * 100}%`,
              background: getStateColor(state, pack),
              opacity: state === currentState ? 1 : 0.35,
              transition: 'opacity 0.3s ease',
            }}
            title={`${state}: ${pct}%`}
          />
        ))}
      </div>
      <div className="flex justify-between">
        <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
          {entries.length} states
        </span>
        <span className="text-[10px] font-medium" style={{ color: getStateColor(currentState, pack) }}>
          {currentState} ({distribution[currentState]}%)
        </span>
      </div>
    </div>
  );
}

/* ─────────────────────────── Live State Pulse ─────────────────────────── */

function LiveStatePulse({ state, label, pack }: { state: string; label: string; pack: Pack }) {
  const color = getStateColor(state, pack);
  return (
    <div className="flex items-center gap-2">
      <div className="relative">
        <div
          className="w-2.5 h-2.5 rounded-full"
          style={{ background: color }}
        />
        <div
          className="w-2.5 h-2.5 rounded-full absolute top-0 left-0"
          style={{
            background: color,
            animation: 'pulse-ring 2s ease-out infinite',
          }}
        />
      </div>
      <span className="text-xs font-mono font-semibold" style={{ color }}>{state}</span>
      <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{label}</span>
    </div>
  );
}

/* ─────────────────────────── Preset Chips ─────────────────────────── */

const PRESETS: Record<string, Record<string, number>> = {
  Scalping: { short_period: 5, mid_period: 13, long_period: 50, fast_period: 8, slow_period: 17, signal_period: 9, sma_period: 10, high_threshold: 2.0, extreme_threshold: 4.0 },
  Swing: { short_period: 9, mid_period: 21, long_period: 200, fast_period: 12, slow_period: 26, signal_period: 9, sma_period: 20, high_threshold: 1.5, extreme_threshold: 3.0 },
  Position: { short_period: 20, mid_period: 50, long_period: 200, fast_period: 19, slow_period: 39, signal_period: 9, sma_period: 50, high_threshold: 1.2, extreme_threshold: 2.5 },
};

/* ─────────────────────────── State Ring Chart (Donut) ─────────────────────────── */

function StateDonut({ distribution, currentState, pack, size = 120 }: { distribution: Record<string, number>; currentState: string; pack: Pack; size?: number }) {
  const entries = Object.entries(distribution);
  const total = entries.reduce((a, [, v]) => a + v, 0);
  const r = (size - 12) / 2;
  const c = size / 2;
  const circumference = 2 * Math.PI * r;

  let offset = 0;
  const arcs = entries.map(([state, pct]) => {
    const len = (pct / total) * circumference;
    const arc = { state, pct, offset, len };
    offset += len;
    return arc;
  });

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ transform: 'rotate(-90deg)' }}>
        {arcs.map((arc) => (
          <circle
            key={arc.state}
            cx={c}
            cy={c}
            r={r}
            fill="none"
            stroke={getStateColor(arc.state, pack)}
            strokeWidth={arc.state === currentState ? 10 : 6}
            strokeDasharray={`${arc.len} ${circumference - arc.len}`}
            strokeDashoffset={-arc.offset}
            opacity={arc.state === currentState ? 1 : 0.3}
            style={{ transition: 'all 0.5s ease' }}
          />
        ))}
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-lg font-bold font-mono" style={{ color: getStateColor(currentState, pack) }}>
          {currentState}
        </span>
        <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{distribution[currentState]}%</span>
      </div>
    </div>
  );
}

/* ─────────────────────────── Parameter Slider ─────────────────────────── */

function ParamSlider({ param, onChange }: { param: PackParam; onChange: (val: number) => void }) {
  const min = Math.max(1, Math.floor(param.value * 0.2));
  const max = Math.ceil(param.value * 3);
  const pct = ((param.value - min) / (max - min)) * 100;

  return (
    <div className="flex items-center gap-3">
      <label className="text-xs w-20 text-right" style={{ color: 'var(--text-muted)' }}>{param.label}</label>
      <div className="flex-1 relative h-8 flex items-center">
        <div className="absolute inset-x-0 h-1.5 rounded-full" style={{ background: 'var(--bg-input)' }} />
        <div
          className="absolute left-0 h-1.5 rounded-full"
          style={{ width: `${pct}%`, background: 'var(--accent)', transition: 'width 0.15s ease' }}
        />
        <input
          type="range"
          min={min}
          max={max}
          step={param.value < 5 ? 0.1 : 1}
          value={param.value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="absolute inset-x-0 w-full h-full opacity-0 cursor-pointer"
          style={{ zIndex: 2 }}
        />
        <div
          className="absolute w-5 h-5 rounded-full border-2 shadow-lg"
          style={{
            left: `calc(${pct}% - 10px)`,
            background: 'var(--bg-card)',
            borderColor: 'var(--accent)',
            transition: 'left 0.15s ease',
            pointerEvents: 'none',
          }}
        />
      </div>
      <span
        className="text-sm font-mono w-12 text-center rounded-md py-0.5"
        style={{ background: 'var(--bg-input)', color: 'var(--text-primary)' }}
      >
        {param.value}
      </span>
    </div>
  );
}

/* ─────────────────────────── Comparison Panel ─────────────────────────── */

function ComparisonPanel({ packA, packB, onClose }: { packA: Pack; packB: Pack; onClose: () => void }) {
  return (
    <Modal title={`${packA.name} (${packA.version}) vs ${packB.name} (${packB.version})`} isOpen onClose={onClose} width="900px">
      <div className="grid grid-cols-2 gap-6">
        {[packA, packB].map((p) => (
          <div key={p.id} className="space-y-4">
            {/* Header */}
            <div className="flex items-center gap-3 pb-3 border-b" style={{ borderColor: 'var(--border)' }}>
              <div
                className="w-14 h-14 rounded-xl flex items-center justify-center"
                style={{ background: 'var(--bg-input)' }}
              >
                {getPackIcon(p, 48)}
              </div>
              <div>
                <h4 className="font-semibold text-sm">{p.name}</h4>
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{p.version}</span>
              </div>
            </div>

            {/* Live state */}
            <LiveStatePulse state={p.currentState} label={p.currentStateLabel} pack={p} />

            {/* Donut */}
            <div className="flex justify-center">
              <StateDonut distribution={p.stateDistribution} currentState={p.currentState} pack={p} size={100} />
            </div>

            {/* Params */}
            <div className="space-y-1">
              <h5 className="text-xs font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Parameters</h5>
              {p.params.map((pr) => (
                <div key={pr.key} className="flex justify-between text-xs px-2 py-1 rounded" style={{ background: 'var(--bg-input)' }}>
                  <span style={{ color: 'var(--text-muted)' }}>{pr.label}</span>
                  <span className="font-mono" style={{ color: 'var(--text-primary)' }}>{pr.value}</span>
                </div>
              ))}
            </div>

            {/* Triggers */}
            <div className="space-y-1">
              <h5 className="text-xs font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Triggers ({p.triggers.length})</h5>
              {p.triggers.map((t) => (
                <div key={t.name} className="flex items-center justify-between text-xs px-2 py-1 rounded" style={{ background: 'var(--bg-input)' }}>
                  <span style={{ color: 'var(--text-primary)' }}>{t.name}</span>
                  <span style={{ color: directionColors[t.direction]?.color }}>{directionColors[t.direction]?.icon} {t.direction}</span>
                </div>
              ))}
            </div>

            {/* Stats */}
            <div className="grid grid-cols-2 gap-2">
              <div className="px-3 py-2 rounded-lg text-center" style={{ background: 'var(--bg-input)' }}>
                <span className="text-lg font-bold" style={{ color: 'var(--accent)' }}>{p.strategiesUsing}</span>
                <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>strategies</p>
              </div>
              <div className="px-3 py-2 rounded-lg text-center" style={{ background: 'var(--bg-input)' }}>
                <span className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>{p.lastTriggered}</span>
                <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>last signal</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </Modal>
  );
}

/* ─────────────────────────── Pack Detail Modal ─────────────────────────── */

function PackDetailModal({ pack, onClose, onUpdateParams }: { pack: Pack; onClose: () => void; onUpdateParams: (id: string, params: PackParam[]) => void }) {
  const [localParams, setLocalParams] = useState<PackParam[]>(pack.params);
  const [activePreset, setActivePreset] = useState<string | null>(null);
  const [detailTab, setDetailTab] = useState<'overview' | 'params' | 'triggers' | 'states'>('overview');

  const applyPreset = (presetName: string) => {
    const preset = PRESETS[presetName];
    const updated = localParams.map((p) => ({
      ...p,
      value: preset[p.key] !== undefined ? preset[p.key] : p.value,
    }));
    setLocalParams(updated);
    setActivePreset(presetName);
  };

  const updateParam = (key: string, val: number) => {
    setLocalParams((prev) => prev.map((p) => (p.key === key ? { ...p, value: val } : p)));
    setActivePreset(null);
  };

  return (
    <Modal title="" isOpen onClose={onClose} width="800px">
      {/* Custom header with icon */}
      <div className="flex items-start gap-4 -mt-2 mb-6">
        <div
          className="w-20 h-20 rounded-2xl flex items-center justify-center flex-shrink-0"
          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
        >
          {getPackIcon(pack, 56)}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <h2 className="text-xl font-bold">{pack.name}</h2>
            <span
              className="text-xs px-2 py-0.5 rounded-full"
              style={{
                color: categoryColors[pack.category]?.color,
                background: categoryColors[pack.category]?.bg,
              }}
            >
              {pack.category}
            </span>
            {pack.isDefault && (
              <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ color: 'var(--text-muted)', background: 'var(--bg-input)' }}>
                DEFAULT
              </span>
            )}
          </div>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>{pack.version} configuration</p>
          <div className="flex items-center gap-4 mt-2">
            <LiveStatePulse state={pack.currentState} label={pack.currentStateLabel} pack={pack} />
          </div>
        </div>
      </div>

      {/* Tab pills */}
      <div className="flex gap-1 mb-5 p-1 rounded-xl" style={{ background: 'var(--bg-input)' }}>
        {(['overview', 'params', 'triggers', 'states'] as const).map((tab) => (
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
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {detailTab === 'overview' && (
        <div className="space-y-5">
          {/* Stats row */}
          <div className="grid grid-cols-3 gap-3">
            <MetricCard label="Strategies Using" value={String(pack.strategiesUsing)} delta={pack.strategiesUsing > 0 ? 'Active' : 'Unused'} positive={pack.strategiesUsing > 0} />
            <MetricCard label="Last Triggered" value={pack.lastTriggered} delta={pack.lastTriggered === 'never' ? 'No signals yet' : 'Recent'} positive={pack.lastTriggered !== 'never'} />
            <MetricCard label="Trigger Count" value={String(pack.triggers.length)} />
          </div>

          {/* Visual preview + donut side by side */}
          <div className="grid grid-cols-2 gap-4">
            <div
              className="rounded-xl p-6 flex items-center justify-center"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
            >
              {getPackIcon(pack, 120)}
            </div>
            <div className="flex items-center justify-center">
              <StateDonut distribution={pack.stateDistribution} currentState={pack.currentState} pack={pack} size={140} />
            </div>
          </div>

          {/* State distribution bar */}
          <div>
            <h4 className="text-xs font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>State Distribution</h4>
            <StateDistributionBar distribution={pack.stateDistribution} currentState={pack.currentState} pack={pack} />
          </div>

          {/* Output badges */}
          <div>
            <h4 className="text-xs font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Interpreter Outputs</h4>
            <div className="flex flex-wrap gap-1.5">
              {pack.outputs.map((output) => {
                const isActive = output === pack.currentState;
                return (
                  <span
                    key={output}
                    className="text-xs font-mono px-2.5 py-1 rounded-lg transition-all"
                    style={{
                      background: isActive ? getStateColor(output, pack) : 'var(--bg-input)',
                      color: isActive ? 'white' : 'var(--text-secondary)',
                      boxShadow: isActive ? `0 0 12px ${getStateColor(output, pack)}40` : 'none',
                      border: isActive ? 'none' : '1px solid var(--border)',
                    }}
                  >
                    {output}
                  </span>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {detailTab === 'params' && (
        <div className="space-y-5">
          {/* Preset buttons */}
          <div>
            <h4 className="text-xs font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Quick Presets</h4>
            <div className="flex gap-2">
              {Object.keys(PRESETS).map((preset) => (
                <button
                  key={preset}
                  onClick={() => applyPreset(preset)}
                  className="px-4 py-2 rounded-lg text-xs font-medium transition-all"
                  style={{
                    background: activePreset === preset ? 'var(--accent)' : 'var(--bg-input)',
                    color: activePreset === preset ? 'white' : 'var(--text-secondary)',
                    border: activePreset === preset ? 'none' : '1px solid var(--border)',
                    boxShadow: activePreset === preset ? '0 0 12px var(--accent-muted)' : 'none',
                  }}
                >
                  {preset}
                </button>
              ))}
            </div>
          </div>

          {/* Visual sliders */}
          <div className="space-y-3">
            <h4 className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>Parameters</h4>
            {localParams.map((param) => (
              <ParamSlider key={param.key} param={param} onChange={(val) => updateParam(param.key, val)} />
            ))}
          </div>

          {/* Live preview of indicator shape */}
          <div
            className="rounded-xl p-6 flex items-center justify-center"
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
          >
            {getPackIcon({ ...pack, params: localParams }, 120)}
          </div>

          {/* Save */}
          <div className="flex gap-3">
            <button
              className="px-4 py-2 rounded-lg text-sm font-medium flex-1"
              style={{ background: 'var(--accent)', color: 'white' }}
              onClick={() => {
                onUpdateParams(pack.id, localParams);
                onClose();
              }}
            >
              Save Parameters
            </button>
            <button
              className="px-4 py-2 rounded-lg text-sm"
              style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
              onClick={() => {
                setLocalParams(pack.params);
                setActivePreset(null);
              }}
            >
              Reset
            </button>
          </div>
        </div>
      )}

      {detailTab === 'triggers' && (
        <div className="space-y-2">
          {pack.triggers.map((trigger, i) => {
            const dir = directionColors[trigger.direction];
            return (
              <div
                key={i}
                className="flex items-center gap-3 px-4 py-3 rounded-xl transition-all"
                style={{
                  background: 'var(--bg-input)',
                  border: '1px solid var(--border)',
                }}
              >
                {/* Direction arrow */}
                <div
                  className="w-8 h-8 rounded-lg flex items-center justify-center text-base"
                  style={{ background: `${dir?.color}15`, color: dir?.color }}
                >
                  {dir?.icon}
                </div>

                <div className="flex-1">
                  <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>{trigger.name}</span>
                  <div className="flex items-center gap-2 mt-0.5">
                    <span className="text-[10px]" style={{ color: dir?.color }}>{trigger.direction}</span>
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded" style={{ background: 'var(--green-muted)', color: 'var(--green)' }}>{trigger.exec}</span>
                  </div>
                </div>

                {/* Simulated fire frequency */}
                <div className="text-right">
                  <span className="text-xs font-mono" style={{ color: 'var(--text-secondary)' }}>
                    {Math.floor(Math.random() * 20 + 5)}/day
                  </span>
                  <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>avg freq</p>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {detailTab === 'states' && (
        <div className="space-y-4">
          <div className="flex justify-center mb-2">
            <StateDonut distribution={pack.stateDistribution} currentState={pack.currentState} pack={pack} size={160} />
          </div>
          <div className="space-y-1.5">
            {Object.entries(pack.stateDistribution)
              .sort(([, a], [, b]) => b - a)
              .map(([state, pct]) => {
                const total = Object.values(pack.stateDistribution).reduce((a, b) => a + b, 0);
                const widthPct = (pct / total) * 100;
                const isActive = state === pack.currentState;
                return (
                  <div
                    key={state}
                    className="flex items-center gap-3 px-3 py-2 rounded-lg transition-all"
                    style={{
                      background: isActive ? `${getStateColor(state, pack)}10` : 'var(--bg-input)',
                      border: isActive ? `1px solid ${getStateColor(state, pack)}40` : '1px solid transparent',
                    }}
                  >
                    <span className="text-xs font-mono w-12" style={{ color: getStateColor(state, pack) }}>{state}</span>
                    <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ background: 'var(--bg-primary)' }}>
                      <div
                        className="h-full rounded-full transition-all"
                        style={{
                          width: `${widthPct}%`,
                          background: getStateColor(state, pack),
                          opacity: isActive ? 1 : 0.5,
                        }}
                      />
                    </div>
                    <span className="text-xs font-mono w-8 text-right" style={{ color: 'var(--text-secondary)' }}>{pct}%</span>
                    {isActive && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded-full" style={{ background: getStateColor(state, pack), color: 'white' }}>
                        LIVE
                      </span>
                    )}
                  </div>
                );
              })}
          </div>
        </div>
      )}
    </Modal>
  );
}

/* ═══════════════════════════ Main Component ═══════════════════════════ */

export default function TfConfluenceV4() {
  const [packs, setPacks] = useState<Pack[]>(initialPacks);
  const [viewMode, setViewMode] = useState<'cards' | 'compact'>('cards');
  const [groupBy, setGroupBy] = useState<'category' | 'usage'>('category');
  const [detailPack, setDetailPack] = useState<Pack | null>(null);
  const [compareSelection, setCompareSelection] = useState<string[]>([]);
  const [showCompare, setShowCompare] = useState(false);
  const [showNewModal, setShowNewModal] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [draggedId, setDraggedId] = useState<string | null>(null);
  const [dragOverId, setDragOverId] = useState<string | null>(null);

  // Drag-and-drop reorder
  const handleDragStart = useCallback((id: string) => setDraggedId(id), []);
  const handleDragOver = useCallback((e: React.DragEvent, id: string) => {
    e.preventDefault();
    setDragOverId(id);
  }, []);
  const handleDrop = useCallback((targetId: string) => {
    if (!draggedId || draggedId === targetId) { setDraggedId(null); setDragOverId(null); return; }
    setPacks((prev) => {
      const arr = [...prev];
      const fromIdx = arr.findIndex((p) => p.id === draggedId);
      const toIdx = arr.findIndex((p) => p.id === targetId);
      const [moved] = arr.splice(fromIdx, 1);
      arr.splice(toIdx, 0, moved);
      return arr;
    });
    setDraggedId(null);
    setDragOverId(null);
  }, [draggedId]);
  const handleDragEnd = useCallback(() => { setDraggedId(null); setDragOverId(null); }, []);

  // Toggle pack enabled
  const togglePack = useCallback((id: string) => {
    setPacks((prev) => prev.map((p) => (p.id === id ? { ...p, enabled: !p.enabled } : p)));
  }, []);

  // Update pack params from detail modal
  const updatePackParams = useCallback((id: string, params: PackParam[]) => {
    setPacks((prev) => prev.map((p) => (p.id === id ? { ...p, params } : p)));
  }, []);

  // Compare toggle
  const toggleCompare = useCallback((id: string) => {
    setCompareSelection((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id);
      if (prev.length >= 2) return [prev[1], id];
      return [...prev, id];
    });
  }, []);

  // Computed
  const enabledCount = packs.filter((p) => p.enabled).length;
  const totalStrategies = new Set(packs.flatMap((p) => Array.from({ length: p.strategiesUsing }, (_, i) => `${p.id}-${i}`))).size;
  const mostUsed = [...packs].sort((a, b) => b.strategiesUsing - a.strategiesUsing)[0];
  const recentlyTriggered = [...packs].filter((p) => p.lastTriggered !== 'never').sort((a, b) => {
    const timeToMin = (t: string) => {
      if (t.includes('m ago')) return parseInt(t);
      if (t.includes('h ago')) return parseInt(t) * 60;
      return 9999;
    };
    return timeToMin(a.lastTriggered) - timeToMin(b.lastTriggered);
  })[0];

  // Filter
  const filtered = packs.filter((p) => {
    if (!searchQuery) return true;
    const q = searchQuery.toLowerCase();
    return p.name.toLowerCase().includes(q) || p.version.toLowerCase().includes(q) || p.category.toLowerCase().includes(q);
  });

  // Group
  const grouped = useMemo(() => {
    const map = new Map<string, Pack[]>();
    if (groupBy === 'category') {
      filtered.forEach((p) => {
        const key = p.category;
        if (!map.has(key)) map.set(key, []);
        map.get(key)!.push(p);
      });
    } else {
      const active = filtered.filter((p) => p.strategiesUsing > 0);
      const unused = filtered.filter((p) => p.strategiesUsing === 0);
      if (active.length) map.set('Active in Strategies', active);
      if (unused.length) map.set('Available but Unused', unused);
    }
    return map;
  }, [filtered, groupBy]);

  // Comparison packs
  const compareA = packs.find((p) => p.id === compareSelection[0]);
  const compareB = packs.find((p) => p.id === compareSelection[1]);

  return (
    <div>
      {/* Pulse animation keyframes */}
      <style>{`
        @keyframes pulse-ring {
          0% { transform: scale(1); opacity: 0.6; }
          100% { transform: scale(2.5); opacity: 0; }
        }
        @keyframes shimmer {
          0% { background-position: -200% 0; }
          100% { background-position: 200% 0; }
        }
        @keyframes fade-in-up {
          0% { opacity: 0; transform: translateY(8px); }
          100% { opacity: 1; transform: translateY(0); }
        }
        .pack-card { animation: fade-in-up 0.4s ease forwards; }
        .pack-card:nth-child(1) { animation-delay: 0.0s; }
        .pack-card:nth-child(2) { animation-delay: 0.05s; }
        .pack-card:nth-child(3) { animation-delay: 0.1s; }
        .pack-card:nth-child(4) { animation-delay: 0.15s; }
        .pack-card:nth-child(5) { animation-delay: 0.2s; }
        .pack-card:nth-child(6) { animation-delay: 0.25s; }
        .drag-over { outline: 2px dashed var(--accent) !important; outline-offset: -2px; }
        .dragging { opacity: 0.4; transform: scale(0.97); }
      `}</style>

      {/* Header */}
      <PageHeader
        title="TF Confluence Packs"
        subtitle="Timeframe-specific indicators -- configure, compare, and monitor live state"
        actions={
          <div className="flex gap-2">
            {compareSelection.length === 2 && (
              <button
                onClick={() => setShowCompare(true)}
                className="px-4 py-2 rounded-lg text-sm font-medium transition-all"
                style={{
                  background: 'var(--purple)',
                  color: 'white',
                  boxShadow: '0 0 16px var(--purple)',
                }}
              >
                Compare ({compareSelection.length})
              </button>
            )}
            <button
              onClick={() => setShowNewModal(true)}
              className="px-4 py-2 rounded-lg text-sm font-medium"
              style={{ background: 'var(--accent)', color: 'white' }}
            >
              + New Pack
            </button>
          </div>
        }
      />

      {/* Summary Dashboard */}
      <div className="grid grid-cols-4 gap-3 mb-6">
        <MetricCard label="Total Packs" value={String(packs.length)} delta={`${enabledCount} enabled`} positive />
        <MetricCard label="Strategy Coverage" value={`${totalStrategies}`} delta="strategies linked" positive />
        <MetricCard label="Most Used" value={mostUsed?.name || '-'} delta={`${mostUsed?.strategiesUsing || 0} strategies`} positive />
        <MetricCard label="Most Recent Signal" value={recentlyTriggered?.name || '-'} delta={recentlyTriggered?.lastTriggered || '-'} positive />
      </div>

      {/* Toolbar */}
      <div className="flex items-center gap-3 mb-5">
        {/* Search */}
        <div className="relative flex-1 max-w-xs">
          <svg
            className="absolute left-3 top-1/2 -translate-y-1/2"
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="var(--text-muted)"
            strokeWidth="2"
            strokeLinecap="round"
          >
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <input
            type="text"
            placeholder="Search packs..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-3 py-2 rounded-lg text-sm"
            style={{
              background: 'var(--bg-input)',
              border: '1px solid var(--border)',
              color: 'var(--text-primary)',
            }}
          />
        </div>

        {/* Group toggle */}
        <div className="flex p-0.5 rounded-lg" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
          {(['category', 'usage'] as const).map((g) => (
            <button
              key={g}
              onClick={() => setGroupBy(g)}
              className="px-3 py-1.5 text-xs rounded-md transition-all"
              style={{
                background: groupBy === g ? 'var(--bg-card)' : 'transparent',
                color: groupBy === g ? 'var(--text-primary)' : 'var(--text-muted)',
                boxShadow: groupBy === g ? '0 1px 3px rgba(0,0,0,0.15)' : 'none',
              }}
            >
              {g === 'category' ? 'By Category' : 'By Usage'}
            </button>
          ))}
        </div>

        {/* View toggle */}
        <div className="flex p-0.5 rounded-lg" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
          {(['cards', 'compact'] as const).map((v) => (
            <button
              key={v}
              onClick={() => setViewMode(v)}
              className="px-3 py-1.5 text-xs rounded-md transition-all"
              style={{
                background: viewMode === v ? 'var(--bg-card)' : 'transparent',
                color: viewMode === v ? 'var(--text-primary)' : 'var(--text-muted)',
                boxShadow: viewMode === v ? '0 1px 3px rgba(0,0,0,0.15)' : 'none',
              }}
            >
              {v === 'cards' ? (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="3" y="3" width="7" height="7" rx="1" />
                  <rect x="14" y="3" width="7" height="7" rx="1" />
                  <rect x="3" y="14" width="7" height="7" rx="1" />
                  <rect x="14" y="14" width="7" height="7" rx="1" />
                </svg>
              ) : (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="3" y1="6" x2="21" y2="6" />
                  <line x1="3" y1="12" x2="21" y2="12" />
                  <line x1="3" y1="18" x2="21" y2="18" />
                </svg>
              )}
            </button>
          ))}
        </div>

        {/* Compare hint */}
        <span className="text-[10px] ml-auto" style={{ color: 'var(--text-muted)' }}>
          {compareSelection.length === 0
            ? 'Click checkboxes to compare'
            : compareSelection.length === 1
              ? 'Select 1 more to compare'
              : 'Ready to compare'}
        </span>
      </div>

      {/* Pack Groups */}
      <div className="space-y-8">
        {Array.from(grouped.entries()).map(([groupName, groupPacks]) => (
          <div key={groupName}>
            {/* Group header */}
            <div className="flex items-center gap-3 mb-3">
              <div
                className="w-1.5 h-5 rounded-full"
                style={{
                  background: groupBy === 'category'
                    ? categoryColors[groupName]?.color || 'var(--accent)'
                    : groupName === 'Active in Strategies'
                      ? 'var(--green)'
                      : 'var(--text-muted)',
                }}
              />
              <h3 className="text-sm font-semibold" style={{ color: 'var(--text-secondary)' }}>{groupName}</h3>
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{groupPacks.length} pack{groupPacks.length !== 1 ? 's' : ''}</span>
            </div>

            {/* Cards view */}
            {viewMode === 'cards' ? (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {groupPacks.map((pack) => {
                  const isCompareSelected = compareSelection.includes(pack.id);
                  const isDragging = draggedId === pack.id;
                  const isDragOver = dragOverId === pack.id && draggedId !== pack.id;

                  return (
                    <div
                      key={pack.id}
                      className={`pack-card ${isDragging ? 'dragging' : ''} ${isDragOver ? 'drag-over' : ''}`}
                      draggable
                      onDragStart={() => handleDragStart(pack.id)}
                      onDragOver={(e) => handleDragOver(e, pack.id)}
                      onDrop={() => handleDrop(pack.id)}
                      onDragEnd={handleDragEnd}
                      style={{ opacity: isDragging ? 0.4 : 1, transition: 'opacity 0.2s, transform 0.2s' }}
                    >
                      <Card className={`relative overflow-hidden ${!pack.enabled ? 'opacity-60' : ''}`}>
                        {/* Subtle accent gradient at top */}
                        <div
                          className="absolute top-0 left-0 right-0 h-0.5"
                          style={{
                            background: pack.enabled
                              ? `linear-gradient(90deg, ${categoryColors[pack.category]?.color || 'var(--accent)'}, transparent)`
                              : 'var(--border)',
                          }}
                        />

                        <div className="flex gap-4">
                          {/* Compare checkbox */}
                          <button
                            onClick={() => toggleCompare(pack.id)}
                            className="w-5 h-5 rounded border flex items-center justify-center flex-shrink-0 mt-1 transition-all"
                            style={{
                              borderColor: isCompareSelected ? 'var(--purple)' : 'var(--border)',
                              background: isCompareSelected ? 'var(--purple)' : 'transparent',
                            }}
                          >
                            {isCompareSelected && (
                              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3">
                                <polyline points="20 6 9 17 4 12" />
                              </svg>
                            )}
                          </button>

                          {/* Indicator Icon */}
                          <div
                            className="w-16 h-16 rounded-xl flex items-center justify-center flex-shrink-0 cursor-pointer transition-transform hover:scale-105"
                            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
                            onClick={() => setDetailPack(pack)}
                          >
                            {getPackIcon(pack, 48)}
                          </div>

                          {/* Content */}
                          <div className="flex-1 min-w-0">
                            <div className="flex items-start justify-between mb-1.5">
                              <div>
                                <div className="flex items-center gap-2 mb-0.5">
                                  <button
                                    onClick={() => setDetailPack(pack)}
                                    className="font-semibold text-sm hover:underline transition-all"
                                    style={{ color: 'var(--text-primary)' }}
                                  >
                                    {pack.name}
                                  </button>
                                  <span
                                    className="text-[10px] px-1.5 py-0.5 rounded"
                                    style={{
                                      background: categoryColors[pack.category]?.bg,
                                      color: categoryColors[pack.category]?.color,
                                    }}
                                  >
                                    {pack.version}
                                  </span>
                                  {pack.isDefault && (
                                    <span className="text-[10px] px-1 py-0.5 rounded" style={{ color: 'var(--text-muted)', background: 'var(--bg-input)' }}>
                                      DEFAULT
                                    </span>
                                  )}
                                </div>
                                <LiveStatePulse state={pack.currentState} label={pack.currentStateLabel} pack={pack} />
                              </div>

                              {/* Toggle */}
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

                            {/* Stats row */}
                            <div className="flex items-center gap-4 mt-2 text-[11px]" style={{ color: 'var(--text-muted)' }}>
                              <span className="flex items-center gap-1">
                                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                  <path d="M12 2L2 7l10 5 10-5-10-5z" />
                                  <path d="M2 17l10 5 10-5" />
                                  <path d="M2 12l10 5 10-5" />
                                </svg>
                                {pack.strategiesUsing} strategies
                              </span>
                              <span className="flex items-center gap-1">
                                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                  <circle cx="12" cy="12" r="10" />
                                  <polyline points="12 6 12 12 16 14" />
                                </svg>
                                {pack.lastTriggered}
                              </span>
                              <span className="flex items-center gap-1">
                                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                  <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
                                </svg>
                                {pack.triggers.length} triggers
                              </span>
                            </div>

                            {/* State distribution */}
                            <div className="mt-2.5">
                              <StateDistributionBar distribution={pack.stateDistribution} currentState={pack.currentState} pack={pack} />
                            </div>

                            {/* Params summary */}
                            <div className="flex flex-wrap gap-1.5 mt-2.5">
                              {pack.params.map((p) => (
                                <span
                                  key={p.key}
                                  className="text-[10px] font-mono px-1.5 py-0.5 rounded"
                                  style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}
                                >
                                  {p.label}:{p.value}
                                </span>
                              ))}
                            </div>
                          </div>
                        </div>

                        {/* Drag handle indicator */}
                        <div
                          className="absolute bottom-0 left-1/2 -translate-x-1/2 w-8 h-1 rounded-full mb-1 opacity-30"
                          style={{ background: 'var(--text-muted)' }}
                        />
                      </Card>
                    </div>
                  );
                })}
              </div>
            ) : (
              /* Compact view */
              <div className="space-y-1.5">
                {groupPacks.map((pack) => {
                  const isCompareSelected = compareSelection.includes(pack.id);

                  return (
                    <div
                      key={pack.id}
                      className="flex items-center gap-3 px-4 py-2.5 rounded-xl transition-all"
                      style={{
                        background: 'var(--bg-card)',
                        border: '1px solid var(--border)',
                        opacity: pack.enabled ? 1 : 0.5,
                      }}
                      draggable
                      onDragStart={() => handleDragStart(pack.id)}
                      onDragOver={(e) => handleDragOver(e, pack.id)}
                      onDrop={() => handleDrop(pack.id)}
                      onDragEnd={handleDragEnd}
                    >
                      {/* Compare checkbox */}
                      <button
                        onClick={() => toggleCompare(pack.id)}
                        className="w-4 h-4 rounded border flex items-center justify-center flex-shrink-0 transition-all"
                        style={{
                          borderColor: isCompareSelected ? 'var(--purple)' : 'var(--border)',
                          background: isCompareSelected ? 'var(--purple)' : 'transparent',
                        }}
                      >
                        {isCompareSelected && (
                          <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3">
                            <polyline points="20 6 9 17 4 12" />
                          </svg>
                        )}
                      </button>

                      {/* Mini icon */}
                      <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0" style={{ background: 'var(--bg-input)' }}>
                        {getPackIcon(pack, 28)}
                      </div>

                      {/* Name */}
                      <button
                        onClick={() => setDetailPack(pack)}
                        className="text-sm font-medium w-36 text-left hover:underline"
                        style={{ color: 'var(--text-primary)' }}
                      >
                        {pack.name}
                      </button>

                      <span
                        className="text-[10px] px-1.5 py-0.5 rounded"
                        style={{ background: categoryColors[pack.category]?.bg, color: categoryColors[pack.category]?.color }}
                      >
                        {pack.version}
                      </span>

                      {/* Live state */}
                      <div className="flex items-center gap-1.5 w-28">
                        <div className="w-2 h-2 rounded-full" style={{ background: getStateColor(pack.currentState, pack) }} />
                        <span className="text-xs font-mono" style={{ color: getStateColor(pack.currentState, pack) }}>{pack.currentState}</span>
                      </div>

                      {/* Mini distribution */}
                      <div className="flex-1">
                        <div className="flex rounded-full overflow-hidden h-1.5" style={{ background: 'var(--bg-input)' }}>
                          {Object.entries(pack.stateDistribution).map(([state, pct]) => {
                            const total = Object.values(pack.stateDistribution).reduce((a, b) => a + b, 0);
                            return (
                              <div
                                key={state}
                                style={{
                                  width: `${(pct / total) * 100}%`,
                                  background: getStateColor(state, pack),
                                  opacity: state === pack.currentState ? 1 : 0.3,
                                }}
                              />
                            );
                          })}
                        </div>
                      </div>

                      {/* Stats */}
                      <span className="text-[10px] w-16 text-center" style={{ color: 'var(--text-muted)' }}>
                        {pack.strategiesUsing} strats
                      </span>
                      <span className="text-[10px] w-14 text-center" style={{ color: 'var(--text-muted)' }}>
                        {pack.lastTriggered}
                      </span>

                      {/* Toggle */}
                      <button
                        onClick={() => togglePack(pack.id)}
                        className="w-8 h-5 rounded-full relative flex-shrink-0 transition-colors"
                        style={{
                          background: pack.enabled ? 'var(--accent)' : 'var(--bg-input)',
                          border: pack.enabled ? 'none' : '1px solid var(--border)',
                        }}
                      >
                        <div
                          className="w-3.5 h-3.5 rounded-full absolute transition-all"
                          style={{
                            background: pack.enabled ? 'white' : 'var(--text-muted)',
                            left: pack.enabled ? '17px' : '3px',
                            top: '3px',
                          }}
                        />
                      </button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Detail Modal */}
      {detailPack && (
        <PackDetailModal
          pack={detailPack}
          onClose={() => setDetailPack(null)}
          onUpdateParams={updatePackParams}
        />
      )}

      {/* Comparison Modal */}
      {showCompare && compareA && compareB && (
        <ComparisonPanel
          packA={compareA}
          packB={compareB}
          onClose={() => setShowCompare(false)}
        />
      )}

      {/* New Pack Modal */}
      <Modal title="Create New Pack" isOpen={showNewModal} onClose={() => setShowNewModal(false)} width="560px">
        <NewPackForm onClose={() => setShowNewModal(false)} />
      </Modal>
    </div>
  );
}

/* ─────────────────────────── New Pack Form ─────────────────────────── */

function NewPackForm({ onClose }: { onClose: () => void }) {
  const [template, setTemplate] = useState('EMA Stack');
  const [versionName, setVersionName] = useState('');
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);

  const templates = [
    { name: 'EMA Stack', category: 'Moving Averages', desc: 'Three exponential moving averages stacked for trend detection' },
    { name: 'MACD Line', category: 'Momentum', desc: 'MACD and signal line crossovers for momentum shifts' },
    { name: 'MACD Histogram', category: 'Momentum', desc: 'Histogram acceleration/deceleration for timing entries' },
    { name: 'VWAP', category: 'Volume', desc: 'Volume-weighted average price with standard deviation bands' },
    { name: 'Relative Volume', category: 'Volume', desc: 'Current volume vs historical average for participation detection' },
  ];

  const selected = templates.find((t) => t.name === template);

  return (
    <div className="space-y-5">
      {/* Template selection */}
      <div>
        <label className="text-xs font-medium mb-2 block" style={{ color: 'var(--text-secondary)' }}>Indicator Template</label>
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
                  {t.name === 'EMA Stack' && <EmaStackIcon state="SML" size={32} />}
                  {t.name === 'MACD Line' && <MacdLineIcon size={32} />}
                  {t.name === 'MACD Histogram' && <MacdHistIcon size={32} />}
                  {t.name === 'VWAP' && <VwapIcon size={32} />}
                  {t.name === 'Relative Volume' && <RvolIcon size={32} />}
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

      {/* Version name */}
      <div>
        <label className="text-xs font-medium mb-1 block" style={{ color: 'var(--text-secondary)' }}>Version Name</label>
        <input
          className="w-full px-3 py-2 rounded-lg text-sm"
          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
          placeholder="e.g. Scalping, Swing, Conservative..."
          value={versionName}
          onChange={(e) => setVersionName(e.target.value)}
        />
      </div>

      {/* Starting preset */}
      <div>
        <label className="text-xs font-medium mb-2 block" style={{ color: 'var(--text-secondary)' }}>Start from Preset</label>
        <div className="flex gap-2">
          {Object.keys(PRESETS).map((preset) => (
            <button
              key={preset}
              onClick={() => setSelectedPreset(selectedPreset === preset ? null : preset)}
              className="flex-1 py-2 rounded-lg text-xs font-medium transition-all"
              style={{
                background: selectedPreset === preset ? 'var(--accent)' : 'var(--bg-input)',
                color: selectedPreset === preset ? 'white' : 'var(--text-secondary)',
                border: selectedPreset === preset ? 'none' : '1px solid var(--border)',
              }}
            >
              {preset}
            </button>
          ))}
        </div>
      </div>

      {/* Preview */}
      <div
        className="rounded-xl p-4 flex items-center gap-4"
        style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
      >
        <div className="w-12 h-12 rounded-lg flex items-center justify-center" style={{ background: 'var(--bg-primary)' }}>
          {selected?.name === 'EMA Stack' && <EmaStackIcon state="SML" size={36} />}
          {selected?.name === 'MACD Line' && <MacdLineIcon size={36} />}
          {selected?.name === 'MACD Histogram' && <MacdHistIcon size={36} />}
          {selected?.name === 'VWAP' && <VwapIcon size={36} />}
          {selected?.name === 'Relative Volume' && <RvolIcon size={36} />}
        </div>
        <div>
          <p className="text-sm font-medium">{selected?.name || template}</p>
          <p className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
            {(selected?.name || template).toLowerCase().replace(/\s+/g, '-')}-{versionName ? versionName.toLowerCase().replace(/\s+/g, '-') : 'custom'}
          </p>
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-3 pt-1">
        <button
          className="px-4 py-2.5 rounded-lg text-sm font-medium flex-1 transition-all"
          style={{ background: 'var(--accent)', color: 'white' }}
          onClick={onClose}
        >
          Create Pack
        </button>
        <button
          className="px-4 py-2.5 rounded-lg text-sm flex-1"
          style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
          onClick={onClose}
        >
          Cancel
        </button>
      </div>
    </div>
  );
}
