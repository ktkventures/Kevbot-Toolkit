'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import Modal from '@/components/Modal';
import ChartPlaceholder from '@/components/ChartPlaceholder';

/* ========================================================================
   Types
   ======================================================================== */

interface ParameterSchema {
  key: string;
  label: string;
  type: 'int' | 'float';
  default: number;
  min: number;
  max: number;
}

interface PlotSetting {
  key: string;
  label: string;
  default: string;
}

interface OutputDef {
  code: string;
  description: string;
}

interface TriggerDef {
  id: string;
  name: string;
  direction: 'LONG' | 'SHORT' | 'BOTH';
  type: 'ENTRY' | 'EXIT';
  execution: string;
}

interface TemplateDef {
  name: string;
  category: string;
  description: string;
  parameters: ParameterSchema[];
  plotSettings: PlotSetting[];
  outputs: OutputDef[];
  triggers: TriggerDef[];
}

interface PackInstance {
  id: string;
  templateKey: string;
  versionName: string;
  enabled: boolean;
  isDefault: boolean;
  paramValues: Record<string, number>;
  plotValues: Record<string, string>;
  lineWidth: number;
  visible: boolean;
}

interface StateTimelineEntry {
  time: string;
  state: string;
  prevState: string;
}

interface TriggerEvent {
  time: string;
  trigger: string;
  direction: string;
  type: string;
  price: string;
  execType: string;
}

/* ========================================================================
   Template Definitions (real data from confluence_groups.py)
   ======================================================================== */

const templates: Record<string, TemplateDef> = {
  ema_stack: {
    name: 'EMA Stack',
    category: 'Moving Averages',
    description: 'Three EMAs for trend direction and momentum',
    parameters: [
      { key: 'short_period', label: 'Short Period', type: 'int', default: 9, min: 1, max: 200 },
      { key: 'mid_period', label: 'Mid Period', type: 'int', default: 21, min: 1, max: 200 },
      { key: 'long_period', label: 'Long Period', type: 'int', default: 200, min: 1, max: 500 },
    ],
    plotSettings: [
      { key: 'short_color', label: 'Short EMA Color', default: '#22c55e' },
      { key: 'mid_color', label: 'Mid EMA Color', default: '#eab308' },
      { key: 'long_color', label: 'Long EMA Color', default: '#ef4444' },
    ],
    outputs: [
      { code: 'SML', description: 'Full Bull Stack -- Short > Mid > Long' },
      { code: 'SLM', description: 'Short leading, Mid dipped below Long' },
      { code: 'MSL', description: 'Mid leading, Short pulling back' },
      { code: 'MLS', description: 'Mid leading, bearish Short' },
      { code: 'LSM', description: 'Long on top, transitional' },
      { code: 'LMS', description: 'Full Bear Stack -- Long > Mid > Short' },
    ],
    triggers: [
      { id: 'ema_cross_bull', name: 'Short > Mid Cross', direction: 'LONG', type: 'ENTRY', execution: '[C]' },
      { id: 'ema_cross_bear', name: 'Short < Mid Cross', direction: 'SHORT', type: 'ENTRY', execution: '[C]' },
      { id: 'ema_mid_cross_bull', name: 'Mid > Long Cross', direction: 'LONG', type: 'ENTRY', execution: '[C]' },
      { id: 'ema_mid_cross_bear', name: 'Mid < Long Cross', direction: 'SHORT', type: 'ENTRY', execution: '[C]' },
    ],
  },
  macd_line: {
    name: 'MACD Line',
    category: 'Momentum',
    description: 'MACD line vs Signal line with zero-line context',
    parameters: [
      { key: 'fast_period', label: 'Fast Period', type: 'int', default: 12, min: 1, max: 100 },
      { key: 'slow_period', label: 'Slow Period', type: 'int', default: 26, min: 1, max: 100 },
      { key: 'signal_period', label: 'Signal Period', type: 'int', default: 9, min: 1, max: 50 },
    ],
    plotSettings: [
      { key: 'macd_color', label: 'MACD Line Color', default: '#2563eb' },
      { key: 'signal_color', label: 'Signal Line Color', default: '#f97316' },
    ],
    outputs: [
      { code: 'M>S+', description: 'MACD above signal, above zero (strong bull)' },
      { code: 'M>S-', description: 'MACD above signal, below zero (recovering)' },
      { code: 'M<S-', description: 'MACD below signal, below zero (strong bear)' },
      { code: 'M<S+', description: 'MACD below signal, above zero (weakening)' },
    ],
    triggers: [
      { id: 'macd_cross_bull', name: 'Bullish Cross', direction: 'LONG', type: 'ENTRY', execution: '[C]' },
      { id: 'macd_cross_bear', name: 'Bearish Cross', direction: 'SHORT', type: 'ENTRY', execution: '[C]' },
      { id: 'macd_zero_cross_up', name: 'Zero Line Cross Up', direction: 'LONG', type: 'ENTRY', execution: '[C]' },
      { id: 'macd_zero_cross_down', name: 'Zero Line Cross Down', direction: 'SHORT', type: 'ENTRY', execution: '[C]' },
    ],
  },
  vwap: {
    name: 'VWAP',
    category: 'Volume',
    description: 'Volume Weighted Average Price with SD bands (7-zone system)',
    parameters: [
      { key: 'sd1_mult', label: 'Inner Band (SD1) Multiplier', type: 'float', default: 1.0, min: 0.5, max: 5.0 },
      { key: 'sd2_mult', label: 'Outer Band (SD2) Multiplier', type: 'float', default: 2.0, min: 0.5, max: 5.0 },
    ],
    plotSettings: [
      { key: 'vwap_color', label: 'VWAP Line Color', default: '#8b5cf6' },
      { key: 'sd1_band_color', label: 'SD1 Band Color', default: '#c4b5fd' },
      { key: 'sd2_band_color', label: 'SD2 Band Color', default: '#ddd6fe' },
    ],
    outputs: [
      { code: '>+2\u03c3', description: 'Price above VWAP + 2xSD (extended high)' },
      { code: '>+1\u03c3', description: 'Price between +1\u03c3 and +2\u03c3' },
      { code: '>V', description: 'Price between VWAP and +1\u03c3 (above VWAP zone)' },
      { code: '@V', description: 'Price within +/-0.5\u03c3 of VWAP (at VWAP)' },
      { code: '<V', description: 'Price between VWAP and -1\u03c3 (below VWAP zone)' },
      { code: '<-1\u03c3', description: 'Price between -1\u03c3 and -2\u03c3' },
      { code: '<-2\u03c3', description: 'Price below VWAP - 2xSD (extended low)' },
    ],
    triggers: [
      { id: 'vwap_cross_above', name: 'Cross Above VWAP', direction: 'LONG', type: 'ENTRY', execution: '[C]' },
      { id: 'vwap_cross_above_ib', name: 'Cross Above VWAP', direction: 'LONG', type: 'ENTRY', execution: '[L0]' },
      { id: 'vwap_cross_below', name: 'Cross Below VWAP', direction: 'SHORT', type: 'ENTRY', execution: '[C]' },
      { id: 'vwap_cross_below_ib', name: 'Cross Below VWAP', direction: 'SHORT', type: 'ENTRY', execution: '[L0]' },
      { id: 'vwap_cross_above_hm', name: 'Cross Above VWAP (HM)', direction: 'LONG', type: 'ENTRY', execution: '[HM]' },
      { id: 'vwap_cross_above_hl', name: 'Cross Above VWAP (HL)', direction: 'LONG', type: 'ENTRY', execution: '[HL]' },
      { id: 'vwap_enter_upper', name: 'Enter Upper Extreme (>+2\u03c3)', direction: 'SHORT', type: 'ENTRY', execution: '[C]' },
      { id: 'vwap_enter_lower', name: 'Enter Lower Extreme (<-2\u03c3)', direction: 'LONG', type: 'ENTRY', execution: '[C]' },
      { id: 'vwap_return', name: 'Return to VWAP Zone', direction: 'BOTH', type: 'EXIT', execution: '[C]' },
    ],
  },
  rvol: {
    name: 'Relative Volume',
    category: 'Volume',
    description: 'Current volume relative to historical average',
    parameters: [
      { key: 'sma_period', label: 'SMA Period', type: 'int', default: 20, min: 5, max: 100 },
      { key: 'high_threshold', label: 'High Threshold', type: 'float', default: 1.5, min: 1.0, max: 5.0 },
      { key: 'extreme_threshold', label: 'Extreme Threshold', type: 'float', default: 3.0, min: 2.0, max: 10.0 },
    ],
    plotSettings: [
      { key: 'bar_color', label: 'Volume Bar Color', default: '#64748b' },
      { key: 'high_color', label: 'High Volume Color', default: '#f59e0b' },
      { key: 'extreme_color', label: 'Extreme Volume Color', default: '#ef4444' },
    ],
    outputs: [
      { code: 'EXTREME', description: 'Volume > 300% of average' },
      { code: 'HIGH', description: 'Volume > 150% of average' },
      { code: 'NORMAL', description: 'Volume 75-150% of average' },
      { code: 'LOW', description: 'Volume 50-75% of average' },
      { code: 'MINIMAL', description: 'Volume < 50% of average' },
    ],
    triggers: [
      { id: 'rvol_spike', name: 'Volume Spike', direction: 'BOTH', type: 'ENTRY', execution: '[C]' },
      { id: 'rvol_extreme', name: 'Extreme Volume', direction: 'BOTH', type: 'ENTRY', execution: '[C]' },
      { id: 'rvol_fade', name: 'Volume Fade', direction: 'BOTH', type: 'EXIT', execution: '[C]' },
    ],
  },
};

/* ========================================================================
   Mock Pack Instances
   ======================================================================== */

const initialPacks: PackInstance[] = [
  {
    id: 'ema-stack-default',
    templateKey: 'ema_stack',
    versionName: 'Default',
    enabled: true,
    isDefault: true,
    paramValues: { short_period: 9, mid_period: 21, long_period: 200 },
    plotValues: { short_color: '#22c55e', mid_color: '#eab308', long_color: '#ef4444' },
    lineWidth: 2,
    visible: true,
  },
  {
    id: 'ema-stack-scalping',
    templateKey: 'ema_stack',
    versionName: 'Scalping',
    enabled: true,
    isDefault: false,
    paramValues: { short_period: 5, mid_period: 13, long_period: 50 },
    plotValues: { short_color: '#22c55e', mid_color: '#eab308', long_color: '#ef4444' },
    lineWidth: 1,
    visible: true,
  },
  {
    id: 'macd-line-default',
    templateKey: 'macd_line',
    versionName: 'Default',
    enabled: true,
    isDefault: true,
    paramValues: { fast_period: 12, slow_period: 26, signal_period: 9 },
    plotValues: { macd_color: '#2563eb', signal_color: '#f97316' },
    lineWidth: 2,
    visible: true,
  },
  {
    id: 'vwap-default',
    templateKey: 'vwap',
    versionName: 'Default',
    enabled: true,
    isDefault: true,
    paramValues: { sd1_mult: 1.0, sd2_mult: 2.0 },
    plotValues: { vwap_color: '#8b5cf6', sd1_band_color: '#c4b5fd', sd2_band_color: '#ddd6fe' },
    lineWidth: 2,
    visible: true,
  },
  {
    id: 'rvol-default',
    templateKey: 'rvol',
    versionName: 'Default',
    enabled: false,
    isDefault: true,
    paramValues: { sma_period: 20, high_threshold: 1.5, extreme_threshold: 3.0 },
    plotValues: { bar_color: '#64748b', high_color: '#f59e0b', extreme_color: '#ef4444' },
    lineWidth: 2,
    visible: true,
  },
];

/* ========================================================================
   Mock Preview Data
   ======================================================================== */

const mockStateTimelines: Record<string, StateTimelineEntry[]> = {
  ema_stack: [
    { time: '09:30:00', state: 'SML', prevState: 'MSL' },
    { time: '09:45:00', state: 'SML', prevState: 'SML' },
    { time: '10:15:00', state: 'MSL', prevState: 'SML' },
    { time: '10:30:00', state: 'MLS', prevState: 'MSL' },
    { time: '11:00:00', state: 'SML', prevState: 'MLS' },
    { time: '11:30:00', state: 'SLM', prevState: 'SML' },
    { time: '12:00:00', state: 'LMS', prevState: 'SLM' },
    { time: '13:15:00', state: 'LSM', prevState: 'LMS' },
    { time: '14:00:00', state: 'SML', prevState: 'LSM' },
  ],
  macd_line: [
    { time: '09:30:00', state: 'M>S+', prevState: 'M<S+' },
    { time: '10:00:00', state: 'M>S-', prevState: 'M>S+' },
    { time: '10:45:00', state: 'M<S-', prevState: 'M>S-' },
    { time: '11:30:00', state: 'M>S-', prevState: 'M<S-' },
    { time: '12:15:00', state: 'M>S+', prevState: 'M>S-' },
  ],
  vwap: [
    { time: '09:30:00', state: '>V', prevState: '@V' },
    { time: '09:45:00', state: '>+1\u03c3', prevState: '>V' },
    { time: '10:15:00', state: '>+2\u03c3', prevState: '>+1\u03c3' },
    { time: '10:45:00', state: '>+1\u03c3', prevState: '>+2\u03c3' },
    { time: '11:00:00', state: '@V', prevState: '>+1\u03c3' },
    { time: '11:30:00', state: '<V', prevState: '@V' },
    { time: '12:00:00', state: '<-1\u03c3', prevState: '<V' },
  ],
  rvol: [
    { time: '09:30:00', state: 'EXTREME', prevState: 'NORMAL' },
    { time: '09:45:00', state: 'HIGH', prevState: 'EXTREME' },
    { time: '10:15:00', state: 'NORMAL', prevState: 'HIGH' },
    { time: '11:00:00', state: 'LOW', prevState: 'NORMAL' },
    { time: '13:00:00', state: 'MINIMAL', prevState: 'LOW' },
  ],
};

const mockTriggerEvents: Record<string, TriggerEvent[]> = {
  ema_stack: [
    { time: '09:31:00', trigger: 'Short > Mid Cross', direction: 'LONG', type: 'ENTRY', price: '$487.23', execType: '[C]' },
    { time: '10:15:00', trigger: 'Short < Mid Cross', direction: 'SHORT', type: 'ENTRY', price: '$486.15', execType: '[C]' },
    { time: '11:01:00', trigger: 'Short > Mid Cross', direction: 'LONG', type: 'ENTRY', price: '$488.02', execType: '[C]' },
    { time: '14:00:00', trigger: 'Mid > Long Cross', direction: 'LONG', type: 'ENTRY', price: '$490.50', execType: '[C]' },
  ],
  macd_line: [
    { time: '09:35:00', trigger: 'Bullish Cross', direction: 'LONG', type: 'ENTRY', price: '$487.80', execType: '[C]' },
    { time: '10:45:00', trigger: 'Bearish Cross', direction: 'SHORT', type: 'ENTRY', price: '$485.90', execType: '[C]' },
    { time: '11:30:00', trigger: 'Bullish Cross', direction: 'LONG', type: 'ENTRY', price: '$486.50', execType: '[C]' },
    { time: '12:15:00', trigger: 'Zero Line Cross Up', direction: 'LONG', type: 'ENTRY', price: '$488.10', execType: '[C]' },
  ],
  vwap: [
    { time: '09:32:00', trigger: 'Cross Above VWAP', direction: 'LONG', type: 'ENTRY', price: '$486.50', execType: '[C]' },
    { time: '10:15:00', trigger: 'Enter Upper Extreme', direction: 'SHORT', type: 'ENTRY', price: '$489.30', execType: '[C]' },
    { time: '11:00:00', trigger: 'Return to VWAP Zone', direction: 'BOTH', type: 'EXIT', price: '$487.00', execType: '[C]' },
    { time: '11:32:00', trigger: 'Cross Below VWAP', direction: 'SHORT', type: 'ENTRY', price: '$486.10', execType: '[L0]' },
    { time: '12:00:00', trigger: 'Enter Lower Extreme', direction: 'LONG', type: 'ENTRY', price: '$484.20', execType: '[C]' },
  ],
  rvol: [
    { time: '09:30:00', trigger: 'Extreme Volume', direction: 'BOTH', type: 'ENTRY', price: '$487.00', execType: '[C]' },
    { time: '09:45:00', trigger: 'Volume Spike', direction: 'BOTH', type: 'ENTRY', price: '$487.80', execType: '[C]' },
    { time: '13:00:00', trigger: 'Volume Fade', direction: 'BOTH', type: 'EXIT', price: '$486.30', execType: '[C]' },
  ],
};

/* ========================================================================
   Style Constants
   ======================================================================== */

const categoryColors: Record<string, { color: string; bg: string }> = {
  'Moving Averages': { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  'Momentum': { color: 'var(--orange)', bg: 'var(--orange-muted)' },
  'Volume': { color: 'var(--accent)', bg: 'var(--accent-muted)' },
  'Volatility': { color: 'var(--red)', bg: 'var(--red-muted)' },
  'Trend': { color: 'var(--green)', bg: 'var(--green-muted)' },
};

const execTypeBadgeStyles: Record<string, { color: string; bg: string }> = {
  '[C]': { color: 'var(--green)', bg: 'var(--green-muted)' },
  '[L0]': { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  '[L1]': { color: 'var(--purple)', bg: 'rgba(168,85,247,0.15)' },
  '[HM]': { color: 'var(--orange)', bg: 'var(--orange-muted)' },
  '[HL]': { color: 'var(--teal)', bg: 'rgba(20,184,166,0.15)' },
};

const directionColors: Record<string, { color: string; bg: string }> = {
  'LONG': { color: 'var(--green)', bg: 'var(--green-muted)' },
  'SHORT': { color: 'var(--red)', bg: 'var(--red-muted)' },
  'BOTH': { color: 'var(--blue)', bg: 'var(--blue-muted)' },
};

const triggerTypeColors: Record<string, { color: string; bg: string }> = {
  'ENTRY': { color: 'var(--accent)', bg: 'var(--accent-muted)' },
  'EXIT': { color: 'var(--orange)', bg: 'var(--orange-muted)' },
};

/* ========================================================================
   Helpers
   ======================================================================== */

function getTemplate(pack: PackInstance): TemplateDef {
  return templates[pack.templateKey];
}

function getPackDisplayName(pack: PackInstance): string {
  const tpl = getTemplate(pack);
  return `${tpl.name} (${pack.versionName})`;
}

function formatParamSummary(pack: PackInstance): string {
  const tpl = getTemplate(pack);
  return tpl.parameters.map((p) => `${p.label}: ${pack.paramValues[p.key]}`).join(', ');
}

function getCategoriesWithPacks(packs: PackInstance[]): { category: string; packs: PackInstance[] }[] {
  const categoryOrder = ['Moving Averages', 'Momentum', 'Volume', 'Volatility', 'Trend'];
  const grouped: Record<string, PackInstance[]> = {};

  for (const pack of packs) {
    const tpl = getTemplate(pack);
    if (!grouped[tpl.category]) grouped[tpl.category] = [];
    grouped[tpl.category].push(pack);
  }

  return categoryOrder
    .filter((cat) => grouped[cat] && grouped[cat].length > 0)
    .map((cat) => ({ category: cat, packs: grouped[cat] }));
}

function getAvailableTemplates(): { key: string; tpl: TemplateDef }[] {
  return Object.entries(templates).map(([key, tpl]) => ({ key, tpl }));
}

/* ========================================================================
   Shared inline-style helpers
   ======================================================================== */

const inputStyle: React.CSSProperties = {
  background: 'var(--bg-input)',
  border: '1px solid var(--border)',
  color: 'var(--text-primary)',
};

const selectStyle: React.CSSProperties = { ...inputStyle };

const btnPrimary: React.CSSProperties = {
  background: 'var(--accent)',
  color: 'white',
};

const btnSecondary: React.CSSProperties = {
  background: 'var(--bg-card)',
  border: '1px solid var(--border)',
  color: 'var(--text-secondary)',
};

/* ========================================================================
   Sub-components
   ======================================================================== */

function ExecBadge({ exec }: { exec: string }) {
  const s = execTypeBadgeStyles[exec] || { color: 'var(--text-muted)', bg: 'var(--bg-input)' };
  return (
    <span
      className="text-xs font-mono px-2 py-0.5 rounded"
      style={{ color: s.color, background: s.bg }}
    >
      {exec}
    </span>
  );
}

function DirectionBadge({ dir }: { dir: string }) {
  const s = directionColors[dir] || { color: 'var(--text-muted)', bg: 'var(--bg-input)' };
  return (
    <span className="text-xs px-2 py-0.5 rounded font-medium" style={{ color: s.color, background: s.bg }}>
      {dir}
    </span>
  );
}

function TypeBadge({ type }: { type: string }) {
  const s = triggerTypeColors[type] || { color: 'var(--text-muted)', bg: 'var(--bg-input)' };
  return (
    <span className="text-xs px-2 py-0.5 rounded" style={{ color: s.color, background: s.bg }}>
      {type}
    </span>
  );
}

function CategoryBadge({ category }: { category: string }) {
  const s = categoryColors[category] || { color: 'var(--text-muted)', bg: 'var(--bg-input)' };
  return (
    <span className="text-xs px-2 py-0.5 rounded font-medium" style={{ color: s.color, background: s.bg }}>
      {category}
    </span>
  );
}

function Toggle({ enabled, onToggle }: { enabled: boolean; onToggle: () => void }) {
  return (
    <button
      onClick={onToggle}
      className="w-10 h-6 rounded-full relative flex-shrink-0 transition-colors"
      style={{
        background: enabled ? 'var(--accent)' : 'var(--bg-input)',
        border: enabled ? 'none' : '1px solid var(--border)',
      }}
    >
      <div
        className="w-4 h-4 rounded-full absolute top-1 transition-all"
        style={{
          background: enabled ? 'white' : 'var(--text-muted)',
          left: enabled ? '22px' : '4px',
        }}
      />
    </button>
  );
}

/* ========================================================================
   Detail: Parameters Tab
   ======================================================================== */

function ParametersTab({ pack, tpl }: { pack: PackInstance; tpl: TemplateDef }) {
  return (
    <Card>
      <div className="flex items-center justify-between mb-5">
        <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
          Pack Parameters
        </h3>
        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
          Template: {tpl.name}
        </span>
      </div>

      <div className="space-y-4">
        {tpl.parameters.map((param) => (
          <div key={param.key}>
            <div className="flex items-center gap-4">
              <label className="text-sm w-52 flex-shrink-0" style={{ color: 'var(--text-secondary)' }}>
                {param.label}
              </label>
              <input
                type="number"
                step={param.type === 'float' ? 0.1 : 1}
                min={param.min}
                max={param.max}
                className="px-3 py-2 rounded-lg text-sm flex-1 font-mono"
                style={inputStyle}
                defaultValue={pack.paramValues[param.key]}
              />
            </div>
            <div className="flex gap-4 mt-1">
              <span className="w-52 flex-shrink-0" />
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                {param.type === 'int' ? 'Integer' : 'Decimal'} &middot; Range: {param.min} - {param.max} &middot; Default: {param.default}
              </span>
            </div>
          </div>
        ))}
      </div>

      <div
        className="flex gap-3 mt-6 pt-4 border-t"
        style={{ borderColor: 'var(--border)' }}
      >
        <button className="px-4 py-2 rounded-lg text-sm font-medium" style={btnPrimary}>
          Save Changes
        </button>
        <button className="px-4 py-2 rounded-lg text-sm" style={btnSecondary}>
          Reset to Default
        </button>
      </div>
    </Card>
  );
}

/* ========================================================================
   Detail: Plot Settings Tab
   ======================================================================== */

function PlotSettingsTab({ pack, tpl }: { pack: PackInstance; tpl: TemplateDef }) {
  const [lineWidth, setLineWidth] = useState(pack.lineWidth);
  const [visible, setVisible] = useState(pack.visible);

  return (
    <Card>
      <h3 className="text-sm font-medium mb-5" style={{ color: 'var(--text-secondary)' }}>
        Visual Settings for {tpl.name}
      </h3>

      {/* Color pickers from template schema */}
      <div className="space-y-4">
        {tpl.plotSettings.map((ps) => (
          <div key={ps.key} className="flex items-center gap-4">
            <label className="text-sm w-52 flex-shrink-0" style={{ color: 'var(--text-secondary)' }}>
              {ps.label}
            </label>
            <div className="flex items-center gap-3">
              <input
                type="color"
                defaultValue={pack.plotValues[ps.key] || ps.default}
                className="w-8 h-8 rounded border-0 cursor-pointer"
                style={{ background: 'transparent', padding: 0 }}
              />
              <div
                className="w-10 h-8 rounded border"
                style={{
                  background: pack.plotValues[ps.key] || ps.default,
                  borderColor: 'var(--border)',
                }}
              />
              <input
                className="px-3 py-2 rounded-lg text-sm font-mono w-28"
                style={inputStyle}
                defaultValue={pack.plotValues[ps.key] || ps.default}
              />
            </div>
          </div>
        ))}

        {/* Divider */}
        <div className="border-t my-2" style={{ borderColor: 'var(--border)' }} />

        {/* Line Width */}
        <div className="flex items-center gap-4">
          <label className="text-sm w-52 flex-shrink-0" style={{ color: 'var(--text-secondary)' }}>
            Line Width
          </label>
          <div className="flex items-center gap-3 flex-1">
            <input
              type="range"
              min="1"
              max="5"
              value={lineWidth}
              onChange={(e) => setLineWidth(Number(e.target.value))}
              className="flex-1"
            />
            <span
              className="text-sm font-mono w-12 text-center px-2 py-1 rounded"
              style={{ background: 'var(--bg-input)', color: 'var(--text-primary)' }}
            >
              {lineWidth}px
            </span>
          </div>
        </div>

        {/* Visible toggle */}
        <div className="flex items-center gap-4">
          <label className="text-sm w-52 flex-shrink-0" style={{ color: 'var(--text-secondary)' }}>
            Visible on Chart
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={visible}
              onChange={(e) => setVisible(e.target.checked)}
              className="w-4 h-4 rounded"
            />
            <span className="text-sm" style={{ color: 'var(--text-primary)' }}>
              {visible ? 'Visible' : 'Hidden'}
            </span>
          </label>
        </div>
      </div>

      <div className="flex gap-3 mt-6 pt-4 border-t" style={{ borderColor: 'var(--border)' }}>
        <button className="px-4 py-2 rounded-lg text-sm font-medium" style={btnPrimary}>
          Save Plot Settings
        </button>
        <button className="px-4 py-2 rounded-lg text-sm" style={btnSecondary}>
          Reset Colors
        </button>
      </div>
    </Card>
  );
}

/* ========================================================================
   Detail: Outputs & Triggers Tab
   ======================================================================== */

function OutputsTriggersTab({ tpl }: { tpl: TemplateDef }) {
  return (
    <div className="grid grid-cols-2 gap-6">
      {/* Outputs column */}
      <Card>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
            Interpreter Outputs
          </h3>
          <span
            className="text-xs px-2 py-0.5 rounded"
            style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}
          >
            {tpl.outputs.length} states
          </span>
        </div>
        <div className="space-y-2">
          {tpl.outputs.map((output) => (
            <div
              key={output.code}
              className="px-3 py-2.5 rounded-lg"
              style={{ background: 'var(--bg-input)' }}
            >
              <div className="flex items-center gap-2 mb-1">
                <span
                  className="text-sm font-mono font-semibold px-2 py-0.5 rounded"
                  style={{ background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}
                >
                  {output.code}
                </span>
                <ExecBadge exec="[C]" />
              </div>
              <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                {output.description}
              </p>
            </div>
          ))}
        </div>
      </Card>

      {/* Triggers column */}
      <Card>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
            Available Triggers
          </h3>
          <span
            className="text-xs px-2 py-0.5 rounded"
            style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}
          >
            {tpl.triggers.length} triggers
          </span>
        </div>
        <div className="space-y-2">
          {tpl.triggers.map((trigger) => (
            <div
              key={trigger.id}
              className="px-3 py-2.5 rounded-lg"
              style={{ background: 'var(--bg-input)' }}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
                  {trigger.name}
                </span>
                <div className="flex items-center gap-1.5">
                  <DirectionBadge dir={trigger.direction} />
                  <TypeBadge type={trigger.type} />
                  <ExecBadge exec={trigger.execution} />
                </div>
              </div>
              <p className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
                {trigger.id}
              </p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

/* ========================================================================
   Detail: Preview Tab
   ======================================================================== */

function PreviewTab({ pack, tpl }: { pack: PackInstance; tpl: TemplateDef }) {
  const [showConditions, setShowConditions] = useState(true);
  const [showTriggers, setShowTriggers] = useState(true);
  const [symbol, setSymbol] = useState('NVDA');
  const [session, setSession] = useState('RTH Only');

  const stateTimeline = mockStateTimelines[pack.templateKey] || [];
  const triggerEvents = mockTriggerEvents[pack.templateKey] || [];

  return (
    <div className="space-y-4">
      {/* Controls bar */}
      <div className="flex items-center gap-3 flex-wrap">
        <select
          className="px-3 py-2 rounded-lg text-sm"
          style={selectStyle}
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
        >
          {['NVDA', 'SPY', 'AAPL', 'TSLA', 'MSFT', 'AMD'].map((s) => (
            <option key={s}>{s}</option>
          ))}
        </select>
        <select className="px-3 py-2 rounded-lg text-sm" style={selectStyle}>
          {['1Min', '5Min', '15Min', '1H', '4H', '1D'].map((tf) => (
            <option key={tf}>{tf}</option>
          ))}
        </select>
        <select
          className="px-3 py-2 rounded-lg text-sm"
          style={selectStyle}
          value={session}
          onChange={(e) => setSession(e.target.value)}
        >
          <option>RTH Only</option>
          <option>Extended Hours</option>
          <option>24/7</option>
        </select>

        <div className="border-l h-6 mx-1" style={{ borderColor: 'var(--border)' }} />

        <label className="flex items-center gap-1.5 cursor-pointer">
          <input
            type="checkbox"
            checked={showConditions}
            onChange={(e) => setShowConditions(e.target.checked)}
            className="w-3.5 h-3.5 rounded"
          />
          <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>Show Conditions</span>
        </label>
        <label className="flex items-center gap-1.5 cursor-pointer">
          <input
            type="checkbox"
            checked={showTriggers}
            onChange={(e) => setShowTriggers(e.target.checked)}
            className="w-3.5 h-3.5 rounded"
          />
          <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>Show Triggers</span>
        </label>
      </div>

      {/* Chart */}
      <Card>
        <ChartPlaceholder
          label={`${symbol} price chart with ${tpl.name} overlay${showConditions ? ' + conditions' : ''}${showTriggers ? ' + triggers' : ''}`}
          height={380}
        />
      </Card>

      {/* Data tables below chart */}
      <div className="grid grid-cols-2 gap-4">
        {/* Interpreter States */}
        <Card>
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
              Interpreter States
            </h4>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
              {stateTimeline.length} transitions
            </span>
          </div>
          <div
            className="rounded-lg overflow-hidden border"
            style={{ borderColor: 'var(--border)' }}
          >
            {/* Header */}
            <div
              className="grid grid-cols-3 text-xs font-medium px-3 py-2"
              style={{ background: 'var(--bg-secondary)', color: 'var(--text-muted)' }}
            >
              <span>Time</span>
              <span>State</span>
              <span>Previous</span>
            </div>
            {/* Rows */}
            {stateTimeline.map((row, i) => {
              const changed = row.state !== row.prevState;
              return (
                <div
                  key={i}
                  className="grid grid-cols-3 text-xs px-3 py-2 border-t"
                  style={{
                    borderColor: 'var(--border)',
                    background: changed ? 'var(--accent-muted)' : 'var(--bg-card)',
                  }}
                >
                  <span className="font-mono" style={{ color: 'var(--text-muted)' }}>
                    {row.time}
                  </span>
                  <span className="font-mono font-semibold" style={{ color: changed ? 'var(--accent)' : 'var(--text-primary)' }}>
                    {row.state}
                  </span>
                  <span className="font-mono" style={{ color: 'var(--text-muted)' }}>
                    {row.prevState}
                  </span>
                </div>
              );
            })}
          </div>
        </Card>

        {/* Trigger Events */}
        <Card>
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
              Trigger Events
            </h4>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
              {triggerEvents.length} events
            </span>
          </div>
          <div
            className="rounded-lg overflow-hidden border"
            style={{ borderColor: 'var(--border)' }}
          >
            {/* Header */}
            <div
              className="grid text-xs font-medium px-3 py-2"
              style={{
                background: 'var(--bg-secondary)',
                color: 'var(--text-muted)',
                gridTemplateColumns: '70px 1fr 55px 50px 70px 35px',
                gap: '4px',
              }}
            >
              <span>Time</span>
              <span>Trigger</span>
              <span>Dir</span>
              <span>Type</span>
              <span>Price</span>
              <span>Exec</span>
            </div>
            {/* Rows */}
            {triggerEvents.map((row, i) => (
              <div
                key={i}
                className="grid text-xs px-3 py-2 border-t items-center"
                style={{
                  borderColor: 'var(--border)',
                  background: 'var(--bg-card)',
                  gridTemplateColumns: '70px 1fr 55px 50px 70px 35px',
                  gap: '4px',
                }}
              >
                <span className="font-mono" style={{ color: 'var(--text-muted)' }}>
                  {row.time}
                </span>
                <span className="font-medium truncate" style={{ color: 'var(--text-primary)' }}>
                  {row.trigger}
                </span>
                <DirectionBadge dir={row.direction} />
                <TypeBadge type={row.type} />
                <span className="font-mono" style={{ color: 'var(--text-primary)' }}>
                  {row.price}
                </span>
                <ExecBadge exec={row.execType} />
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}

/* ========================================================================
   Detail: Code Tab
   ======================================================================== */

function CodeTab({ pack, tpl }: { pack: PackInstance; tpl: TemplateDef }) {
  const [showIndicatorCode, setShowIndicatorCode] = useState(false);
  const [showInterpreterCode, setShowInterpreterCode] = useState(false);
  const [showPineScript, setShowPineScript] = useState(false);

  const codeBlockStyle: React.CSSProperties = {
    background: 'var(--bg-primary)',
    border: '1px solid var(--border)',
    color: 'var(--text-secondary)',
    fontFamily: 'monospace',
    fontSize: '12px',
    lineHeight: '1.6',
    padding: '16px',
    borderRadius: '8px',
    whiteSpace: 'pre-wrap',
    overflowX: 'auto',
  };

  return (
    <div className="space-y-4">
      {/* Active Configuration */}
      <Card>
        <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>
          Active Parameters
        </h3>
        <div
          className="rounded-lg px-4 py-3"
          style={{ background: 'var(--bg-primary)', border: '1px solid var(--border)' }}
        >
          <div className="grid grid-cols-2 gap-y-2 gap-x-8">
            {tpl.parameters.map((p) => (
              <div key={p.key} className="flex justify-between">
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{p.label}</span>
                <span className="text-xs font-mono font-semibold" style={{ color: 'var(--text-primary)' }}>
                  {pack.paramValues[p.key]}
                </span>
              </div>
            ))}
          </div>
        </div>
      </Card>

      {/* Indicator Function */}
      <Card>
        <button
          className="flex items-center justify-between w-full text-left"
          onClick={() => setShowIndicatorCode(!showIndicatorCode)}
        >
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
            Indicator Function
          </h3>
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {showIndicatorCode ? 'Collapse' : 'Expand'}
          </span>
        </button>
        {showIndicatorCode && (
          <div className="mt-3">
            <pre style={codeBlockStyle}>
{`def compute_${pack.templateKey}(df, ${tpl.parameters.map((p) => `${p.key}=${pack.paramValues[p.key]}`).join(', ')}):
    """
    Compute ${tpl.name} indicator values.
    Template: ${tpl.name} (${tpl.category})

    Parameters:
${tpl.parameters.map((p) => `        ${p.key}: ${p.label} (${p.type}, range ${p.min}-${p.max})`).join('\n')}

    Returns enriched DataFrame with indicator columns.
    """
    # Implementation in indicators.py
    ...`}
            </pre>
          </div>
        )}
      </Card>

      {/* Interpreter Function */}
      <Card>
        <button
          className="flex items-center justify-between w-full text-left"
          onClick={() => setShowInterpreterCode(!showInterpreterCode)}
        >
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
            Interpreter Function
          </h3>
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {showInterpreterCode ? 'Collapse' : 'Expand'}
          </span>
        </button>
        {showInterpreterCode && (
          <div className="mt-3">
            <pre style={codeBlockStyle}>
{`def interpret_${pack.templateKey}(row):
    """
    Classify bar state into one of ${tpl.outputs.length} output categories.

    Outputs:
${tpl.outputs.map((o) => `        ${o.code}: ${o.description}`).join('\n')}

    Returns state string for confluence record.
    """
    # Implementation in interpreters.py
    ...`}
            </pre>
          </div>
        )}
      </Card>

      {/* Pine Script Reference */}
      <Card>
        <button
          className="flex items-center justify-between w-full text-left"
          onClick={() => setShowPineScript(!showPineScript)}
        >
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
            Pine Script Reference
          </h3>
          <span className="text-xs px-2 py-0.5 rounded" style={{ color: 'var(--text-muted)', background: 'var(--bg-input)' }}>
            Optional
          </span>
        </button>
        {showPineScript && (
          <div className="mt-3">
            <pre style={codeBlockStyle}>
{`//@version=5
// ${tpl.name} -- Reference implementation
// Parameters: ${tpl.parameters.map((p) => `${p.key}=${pack.paramValues[p.key]}`).join(', ')}
indicator("${tpl.name}", overlay=true)

// Pine Script equivalent for TradingView reference
// This is a read-only reference -- edits here have no effect
...`}
            </pre>
          </div>
        )}
      </Card>
    </div>
  );
}

/* ========================================================================
   Detail: Danger Zone Tab
   ======================================================================== */

function DangerZoneTab({ pack }: { pack: PackInstance }) {
  const [confirmDelete, setConfirmDelete] = useState(false);

  return (
    <Card>
      <h3 className="text-sm font-medium mb-5" style={{ color: 'var(--red)' }}>
        Danger Zone
      </h3>
      <div className="space-y-6">
        {/* Rename */}
        <div>
          <label className="text-sm mb-2 block" style={{ color: 'var(--text-secondary)' }}>
            Rename Version
          </label>
          <div className="flex gap-3">
            <input
              className="px-3 py-2 rounded-lg text-sm flex-1"
              style={inputStyle}
              defaultValue={pack.versionName}
              disabled={pack.isDefault}
            />
            <button
              className="px-4 py-2 rounded-lg text-sm"
              style={{
                ...btnSecondary,
                ...(pack.isDefault ? { color: 'var(--text-muted)', cursor: 'not-allowed' } : {}),
              }}
              disabled={pack.isDefault}
            >
              Rename
            </button>
          </div>
          {pack.isDefault && (
            <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
              Default packs cannot be renamed.
            </p>
          )}
        </div>

        {/* Reset to defaults */}
        <div className="pt-4 border-t" style={{ borderColor: 'var(--border)' }}>
          <p className="text-sm mb-2" style={{ color: 'var(--text-secondary)' }}>
            Reset all parameters and plot settings to template defaults.
          </p>
          <button className="px-4 py-2 rounded-lg text-sm" style={btnSecondary}>
            Reset to Defaults
          </button>
        </div>

        {/* Delete */}
        <div className="pt-4 border-t" style={{ borderColor: 'var(--border)' }}>
          <p className="text-sm mb-2" style={{ color: 'var(--text-secondary)' }}>
            Delete this pack permanently. This cannot be undone.
          </p>
          {!confirmDelete ? (
            <button
              className="px-4 py-2 rounded-lg text-sm font-medium"
              style={{
                background: pack.isDefault ? 'var(--bg-input)' : 'var(--red)',
                color: pack.isDefault ? 'var(--text-muted)' : 'white',
                cursor: pack.isDefault ? 'not-allowed' : 'pointer',
              }}
              disabled={pack.isDefault}
              onClick={() => setConfirmDelete(true)}
            >
              Delete Pack
            </button>
          ) : (
            <div className="flex items-center gap-3">
              <button
                className="px-4 py-2 rounded-lg text-sm font-medium"
                style={{ background: 'var(--red)', color: 'white' }}
              >
                Confirm Delete
              </button>
              <button
                className="px-4 py-2 rounded-lg text-sm"
                style={btnSecondary}
                onClick={() => setConfirmDelete(false)}
              >
                Cancel
              </button>
            </div>
          )}
          {pack.isDefault && (
            <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
              Default packs cannot be deleted.
            </p>
          )}
        </div>
      </div>
    </Card>
  );
}

/* ========================================================================
   Detail View (all 6 tabs)
   ======================================================================== */

function DetailView({
  pack,
  onBack,
}: {
  pack: PackInstance;
  onBack: () => void;
}) {
  const tpl = getTemplate(pack);

  return (
    <div>
      <PageHeader
        title={getPackDisplayName(pack)}
        subtitle={`${tpl.category} -- ${tpl.description}`}
        backHref="#"
        actions={
          <div className="flex items-center gap-3">
            <Toggle enabled={pack.enabled} onToggle={() => {}} />
            <span className="text-xs" style={{ color: pack.enabled ? 'var(--green)' : 'var(--text-muted)' }}>
              {pack.enabled ? 'Enabled' : 'Disabled'}
            </span>
            <button
              onClick={onBack}
              className="px-4 py-2 rounded-lg text-sm"
              style={btnSecondary}
            >
              Back to Packs
            </button>
          </div>
        }
      />

      <TabBar tabs={['Parameters', 'Plot Settings', 'Outputs & Triggers', 'Preview', 'Code', 'Danger Zone']}>
        {(tab) => (
          <div>
            {tab === 'Parameters' && <ParametersTab pack={pack} tpl={tpl} />}
            {tab === 'Plot Settings' && <PlotSettingsTab pack={pack} tpl={tpl} />}
            {tab === 'Outputs & Triggers' && <OutputsTriggersTab tpl={tpl} />}
            {tab === 'Preview' && <PreviewTab pack={pack} tpl={tpl} />}
            {tab === 'Code' && <CodeTab pack={pack} tpl={tpl} />}
            {tab === 'Danger Zone' && <DangerZoneTab pack={pack} />}
          </div>
        )}
      </TabBar>
    </div>
  );
}

/* ========================================================================
   Pack Card (list view row)
   ======================================================================== */

function PackCard({
  pack,
  tpl,
  onToggle,
  onDetail,
  onCopy,
}: {
  pack: PackInstance;
  tpl: TemplateDef;
  onToggle: () => void;
  onDetail: () => void;
  onCopy: () => void;
}) {
  return (
    <Card>
      <div className="flex items-start gap-4">
        {/* Toggle */}
        <div className="pt-0.5">
          <Toggle enabled={pack.enabled} onToggle={onToggle} />
        </div>

        {/* Main info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="font-semibold text-sm" style={{ color: 'var(--text-primary)' }}>
              {tpl.name}
            </span>
            {pack.isDefault && (
              <span
                className="text-xs px-1.5 py-0.5 rounded"
                style={{ color: 'var(--text-muted)', background: 'var(--bg-input)' }}
              >
                default
              </span>
            )}
            {!pack.isDefault && (
              <span
                className="text-xs px-1.5 py-0.5 rounded font-medium"
                style={{ color: 'var(--accent)', background: 'var(--accent-muted)' }}
              >
                {pack.versionName}
              </span>
            )}
          </div>

          {/* Parameter summary */}
          <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>
            {formatParamSummary(pack)}
          </p>

          {/* Outputs preview */}
          <div className="flex items-center gap-1 flex-wrap">
            {tpl.outputs.slice(0, 4).map((output) => (
              <span
                key={output.code}
                className="text-xs font-mono px-1.5 py-0.5 rounded"
                style={{ background: 'var(--bg-input)', color: 'var(--text-secondary)' }}
                title={output.description}
              >
                {output.code}
              </span>
            ))}
            {tpl.outputs.length > 4 && (
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                +{tpl.outputs.length - 4} more
              </span>
            )}
          </div>
        </div>

        {/* Right side: trigger count + actions */}
        <div className="flex flex-col items-end gap-2 flex-shrink-0">
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {tpl.triggers.length} triggers
          </span>
          <div className="flex gap-2">
            <button
              onClick={onDetail}
              className="px-3 py-1.5 rounded text-xs font-medium"
              style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
            >
              Details
            </button>
            <button
              onClick={onCopy}
              className="px-3 py-1.5 rounded text-xs"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
            >
              Copy
            </button>
          </div>
        </div>
      </div>
    </Card>
  );
}

/* ========================================================================
   New Pack Modal
   ======================================================================== */

function NewPackModal({
  isOpen,
  onClose,
  onSubmit,
}: {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (templateKey: string, versionName: string) => void;
}) {
  const [selectedTemplate, setSelectedTemplate] = useState(getAvailableTemplates()[0].key);
  const [versionName, setVersionName] = useState('');

  const tpl = templates[selectedTemplate];

  return (
    <Modal title="Create New Pack" isOpen={isOpen} onClose={onClose} width="560px">
      <div className="space-y-5">
        {/* Template selector */}
        <div>
          <label className="text-sm mb-1.5 block font-medium" style={{ color: 'var(--text-secondary)' }}>
            Template
          </label>
          <select
            className="w-full px-3 py-2 rounded-lg text-sm"
            style={selectStyle}
            value={selectedTemplate}
            onChange={(e) => setSelectedTemplate(e.target.value)}
          >
            {getAvailableTemplates().map(({ key, tpl: t }) => (
              <option key={key} value={key}>
                {t.name} ({t.category})
              </option>
            ))}
          </select>
          <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
            {tpl.description}
          </p>
        </div>

        {/* Template preview */}
        <div
          className="rounded-lg px-4 py-3"
          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
        >
          <div className="flex items-center gap-2 mb-2">
            <CategoryBadge category={tpl.category} />
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
              {tpl.parameters.length} params &middot; {tpl.outputs.length} outputs &middot; {tpl.triggers.length} triggers
            </span>
          </div>
          <div className="flex items-center gap-1 flex-wrap">
            {tpl.outputs.slice(0, 5).map((o) => (
              <span
                key={o.code}
                className="text-xs font-mono px-1.5 py-0.5 rounded"
                style={{ background: 'var(--bg-secondary)', color: 'var(--text-secondary)' }}
              >
                {o.code}
              </span>
            ))}
            {tpl.outputs.length > 5 && (
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                +{tpl.outputs.length - 5}
              </span>
            )}
          </div>
        </div>

        {/* Version name */}
        <div>
          <label className="text-sm mb-1.5 block font-medium" style={{ color: 'var(--text-secondary)' }}>
            Version Name
          </label>
          <input
            className="w-full px-3 py-2 rounded-lg text-sm"
            style={inputStyle}
            placeholder="e.g. Scalping, Swing, Conservative..."
            value={versionName}
            onChange={(e) => setVersionName(e.target.value)}
          />
        </div>

        {/* Default parameters preview */}
        <div>
          <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>
            Default Parameters
          </label>
          <div
            className="rounded-lg px-4 py-2 text-xs font-mono"
            style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}
          >
            {tpl.parameters.map((p) => `${p.label}: ${p.default}`).join('  |  ')}
          </div>
        </div>

        {/* Pack ID Preview */}
        <div>
          <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>
            Pack ID Preview
          </label>
          <p
            className="text-sm font-mono px-3 py-2 rounded-lg"
            style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}
          >
            {selectedTemplate.replace(/_/g, '-')}-{versionName ? versionName.toLowerCase().replace(/\s+/g, '-') : 'custom'}
          </p>
        </div>

        {/* Actions */}
        <div className="flex gap-3 pt-2">
          <button
            className="px-4 py-2 rounded-lg text-sm font-medium flex-1"
            style={btnPrimary}
            onClick={() => {
              onSubmit(selectedTemplate, versionName || 'Custom');
              onClose();
            }}
          >
            Create Pack
          </button>
          <button
            className="px-4 py-2 rounded-lg text-sm flex-1"
            style={btnSecondary}
            onClick={onClose}
          >
            Cancel
          </button>
        </div>
      </div>
    </Modal>
  );
}

/* ========================================================================
   Main Component
   ======================================================================== */

export default function TfConfluenceV2() {
  const [packs, setPacks] = useState<PackInstance[]>(initialPacks);
  const [detailPackId, setDetailPackId] = useState<string | null>(null);
  const [showNewModal, setShowNewModal] = useState(false);

  const enabledCount = packs.filter((p) => p.enabled).length;
  const categorized = getCategoriesWithPacks(packs);

  function togglePack(id: string) {
    setPacks((prev) =>
      prev.map((p) => (p.id === id ? { ...p, enabled: !p.enabled } : p))
    );
  }

  function handleCopyPack(pack: PackInstance) {
    const tpl = getTemplate(pack);
    const newId = `${pack.templateKey.replace(/_/g, '-')}-copy-${Date.now()}`;
    const copy: PackInstance = {
      ...pack,
      id: newId,
      versionName: `${pack.versionName} Copy`,
      isDefault: false,
      paramValues: { ...pack.paramValues },
      plotValues: { ...pack.plotValues },
    };
    setPacks((prev) => [...prev, copy]);
  }

  function handleCreatePack(templateKey: string, versionName: string) {
    const tpl = templates[templateKey];
    const newId = `${templateKey.replace(/_/g, '-')}-${versionName.toLowerCase().replace(/\s+/g, '-')}-${Date.now()}`;
    const paramValues: Record<string, number> = {};
    for (const p of tpl.parameters) paramValues[p.key] = p.default;
    const plotValues: Record<string, string> = {};
    for (const ps of tpl.plotSettings) plotValues[ps.key] = ps.default;

    const newPack: PackInstance = {
      id: newId,
      templateKey,
      versionName,
      enabled: true,
      isDefault: false,
      paramValues,
      plotValues,
      lineWidth: 2,
      visible: true,
    };
    setPacks((prev) => [...prev, newPack]);
  }

  // Detail view
  const detailPack = packs.find((p) => p.id === detailPackId);
  if (detailPack) {
    return <DetailView pack={detailPack} onBack={() => setDetailPackId(null)} />;
  }

  // List view
  return (
    <div>
      <PageHeader
        title="TF Confluence Packs"
        subtitle="Timeframe-specific indicators that compute on each bar"
        actions={
          <button
            onClick={() => setShowNewModal(true)}
            className="px-4 py-2 rounded-lg text-sm font-medium"
            style={btnPrimary}
          >
            + New Pack
          </button>
        }
      />

      {/* Summary stats */}
      <div className="flex items-center gap-4 mb-6">
        <span className="text-sm" style={{ color: 'var(--text-muted)' }}>
          {packs.length} packs, {enabledCount} enabled
        </span>
        <div className="flex items-center gap-1.5">
          {Object.entries(categoryColors)
            .filter(([cat]) => categorized.some((c) => c.category === cat))
            .map(([cat, style]) => (
              <span
                key={cat}
                className="text-xs px-2 py-0.5 rounded"
                style={{ color: style.color, background: style.bg }}
              >
                {cat}
              </span>
            ))}
        </div>
      </div>

      {/* Grouped pack cards */}
      <div className="space-y-6">
        {categorized.map(({ category, packs: catPacks }) => (
          <div key={category}>
            {/* Category header */}
            <div className="flex items-center gap-3 mb-3">
              <CategoryBadge category={category} />
              <div className="flex-1 border-t" style={{ borderColor: 'var(--border)' }} />
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                {catPacks.length} pack{catPacks.length !== 1 ? 's' : ''}
              </span>
            </div>

            {/* Pack cards in this category */}
            <div className="space-y-3">
              {catPacks.map((pack) => {
                const tpl = getTemplate(pack);
                return (
                  <PackCard
                    key={pack.id}
                    pack={pack}
                    tpl={tpl}
                    onToggle={() => togglePack(pack.id)}
                    onDetail={() => setDetailPackId(pack.id)}
                    onCopy={() => handleCopyPack(pack)}
                  />
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {/* New Pack Modal */}
      <NewPackModal
        isOpen={showNewModal}
        onClose={() => setShowNewModal(false)}
        onSubmit={handleCreatePack}
      />
    </div>
  );
}
