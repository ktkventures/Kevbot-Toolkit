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
  type: 'int' | 'float' | 'select';
  default: number | string;
  min?: number;
  max?: number;
  options?: string[];
  help?: string;
  /** For composite template: only show when stop_method matches */
  visibleWhen?: { field: string; value: string };
}

interface TemplateDef {
  name: string;
  category: string;
  description: string;
  parameters: ParameterSchema[];
  stopSummary: string;
  targetSummary: string;
  contextualHelp: string[];
}

interface PackInstance {
  id: string;
  templateKey: string;
  versionName: string;
  enabled: boolean;
  isDefault: boolean;
  paramValues: Record<string, number | string>;
  description: string;
}

interface StopPreviewEntry {
  label: string;
  entryPrice: string;
  stopPrice: string;
  targetPrice: string;
  riskReward: string;
}

/* ========================================================================
   Template Definitions (real data from risk_management_packs.py)
   ======================================================================== */

const templates: Record<string, TemplateDef> = {
  atr_based: {
    name: 'ATR-Based',
    category: 'Volatility',
    description: 'ATR-derived stop and target with independent multipliers',
    parameters: [
      { key: 'stop_atr_mult', label: 'Stop ATR Mult', type: 'float', default: 1.5, min: 0.5, max: 5.0, help: 'ATR multiplier controls how far the stop is placed from entry' },
      { key: 'target_atr_mult', label: 'Target ATR Mult', type: 'float', default: 3.0, min: 0.0, max: 10.0, help: 'Set to 0 for no ATR target (exit via signal or bar count only)' },
    ],
    stopSummary: 'ATR x {stop_atr_mult}',
    targetSummary: 'ATR x {target_atr_mult}',
    contextualHelp: [
      'ATR (Average True Range) measures volatility over the last 14 bars by default.',
      'Higher multipliers give the trade more room to breathe but increase risk per trade.',
      'A 1.5x ATR stop with 3.0x ATR target gives a natural 2:1 reward-to-risk ratio.',
      'ATR stops automatically adapt to market conditions -- wider in volatile markets, tighter in quiet ones.',
    ],
  },

  fixed_dollar: {
    name: 'Fixed Dollar',
    category: 'Fixed',
    description: 'Fixed dollar amounts for stop and target',
    parameters: [
      { key: 'stop_amount', label: 'Stop ($)', type: 'float', default: 1.0, min: 0.01, max: 100.0, help: 'Fixed dollar distance from entry to stop' },
      { key: 'target_amount', label: 'Target ($)', type: 'float', default: 2.0, min: 0.0, max: 100.0, help: 'Fixed dollar distance from entry to target (0 = no target)' },
    ],
    stopSummary: '${stop_amount}',
    targetSummary: '${target_amount}',
    contextualHelp: [
      'Fixed dollar stops use a constant distance regardless of volatility.',
      'Best suited for instruments with stable price ranges (e.g., SPY 1-min scalps).',
      'Warning: Fixed stops may be too tight in volatile conditions or too loose in quiet ones.',
      'Consider ATR-based stops for instruments with variable volatility.',
    ],
  },

  percentage: {
    name: 'Percentage',
    category: 'Fixed',
    description: 'Percentage-based stop and target relative to entry price',
    parameters: [
      { key: 'stop_pct', label: 'Stop (%)', type: 'float', default: 0.5, min: 0.01, max: 10.0, help: 'Percentage distance from entry to stop' },
      { key: 'target_pct', label: 'Target (%)', type: 'float', default: 1.0, min: 0.0, max: 20.0, help: 'Percentage distance from entry to target (0 = no target)' },
    ],
    stopSummary: '{stop_pct}%',
    targetSummary: '{target_pct}%',
    contextualHelp: [
      'Percentage stops scale with the instrument price -- a 0.5% stop on a $100 stock is $0.50.',
      'Works well across different price levels without manual adjustment.',
      'Common for swing trading where absolute dollar amounts vary by instrument.',
    ],
  },

  swing: {
    name: 'Swing',
    category: 'Structure',
    description: 'Swing-based stop with risk:reward target',
    parameters: [
      { key: 'lookback', label: 'Lookback', type: 'int', default: 5, min: 2, max: 50, help: 'Number of bars to search for swing high/low' },
      { key: 'padding', label: 'Padding ($)', type: 'float', default: 0.05, min: 0.0, max: 10.0, help: 'Extra buffer beyond the swing point to avoid false triggers' },
      { key: 'rr_ratio', label: 'R:R Ratio', type: 'float', default: 2.0, min: 0.0, max: 10.0, help: 'Target distance as multiple of stop distance (0 = no target)' },
    ],
    stopSummary: 'Swing ({lookback} bars, ${padding} pad)',
    targetSummary: '{rr_ratio}R',
    contextualHelp: [
      'Places the stop at the recent swing low (for longs) or swing high (for shorts).',
      'Respects market structure -- stops are placed at logical support/resistance levels.',
      'Padding adds a small buffer beyond the swing point to account for noise.',
      'The R:R ratio automatically sizes the target relative to the stop distance.',
      'Stop-validity guard: entries are rejected when swing stop >= entry price (no room for risk).',
    ],
  },

  rr_ratio: {
    name: 'Risk:Reward',
    category: 'Composite',
    description: 'Any stop method paired with a fixed risk:reward target',
    parameters: [
      { key: 'stop_method', label: 'Stop Method', type: 'select', default: 'atr', options: ['atr', 'fixed_dollar', 'percentage', 'swing'] },
      { key: 'stop_atr_mult', label: 'Stop ATR Mult', type: 'float', default: 1.5, min: 0.5, max: 5.0, visibleWhen: { field: 'stop_method', value: 'atr' } },
      { key: 'stop_amount', label: 'Stop ($)', type: 'float', default: 1.0, min: 0.01, max: 100.0, visibleWhen: { field: 'stop_method', value: 'fixed_dollar' } },
      { key: 'stop_pct', label: 'Stop (%)', type: 'float', default: 0.5, min: 0.01, max: 10.0, visibleWhen: { field: 'stop_method', value: 'percentage' } },
      { key: 'lookback', label: 'Lookback', type: 'int', default: 5, min: 2, max: 50, visibleWhen: { field: 'stop_method', value: 'swing' } },
      { key: 'padding', label: 'Padding ($)', type: 'float', default: 0.05, min: 0.0, max: 10.0, visibleWhen: { field: 'stop_method', value: 'swing' } },
      { key: 'rr_ratio', label: 'R:R Ratio', type: 'float', default: 2.0, min: 0.5, max: 10.0, help: 'Target as a multiple of stop risk' },
    ],
    stopSummary: '{stop_method} stop',
    targetSummary: '{rr_ratio}R',
    contextualHelp: [
      'Composite pack: choose any stop method and pair it with a fixed R:R target.',
      'The stop method determines how the stop distance is calculated.',
      'The R:R ratio sizes the target proportionally to whatever stop method is active.',
      'Useful for standardizing risk:reward across different stop calculation methods.',
    ],
  },

  atr_trailing: {
    name: 'ATR Trailing',
    category: 'Trailing',
    description: 'ATR-based initial stop with trailing stop that ratchets in your favor',
    parameters: [
      { key: 'stop_atr_mult', label: 'Initial Stop ATR\u00d7', type: 'float', default: 1.5, min: 0.5, max: 5.0, help: 'Initial stop distance from entry' },
      { key: 'trail_atr_mult', label: 'Trail ATR\u00d7', type: 'float', default: 1.0, min: 0.3, max: 5.0, help: 'Trailing stop ATR distance (typically tighter than initial)' },
      { key: 'trail_activation_r', label: 'Trail Activation (R)', type: 'float', default: 0.5, min: 0.0, max: 5.0, help: 'Profit in R-multiples before trailing begins (0 = immediate)' },
      { key: 'target_atr_mult', label: 'Target ATR\u00d7 (0=none)', type: 'float', default: 0.0, min: 0.0, max: 10.0, help: 'Optional fixed ATR target; 0 means let the trail ride' },
    ],
    stopSummary: 'ATR x{stop_atr_mult} -> Trail x{trail_atr_mult}',
    targetSummary: 'ATR x{target_atr_mult}',
    contextualHelp: [
      'Starts with a standard ATR stop, then transitions to a trailing stop after profit reaches the activation threshold.',
      'The trailing stop ratchets upward (for longs) as price moves in your favor -- it never moves backward.',
      'Set target to 0 for a pure trend-following approach: the trail captures all gains until reversal.',
      'Activation R = 0 means trailing begins immediately on any favorable price movement.',
    ],
  },

  breakeven_stop: {
    name: 'Breakeven Stop',
    category: 'Trailing',
    description: 'ATR stop that moves to breakeven after reaching an R threshold',
    parameters: [
      { key: 'stop_atr_mult', label: 'Initial Stop ATR\u00d7', type: 'float', default: 1.5, min: 0.5, max: 5.0, help: 'Initial stop distance from entry' },
      { key: 'be_activation_r', label: 'BE Activation (R)', type: 'float', default: 1.0, min: 0.1, max: 5.0, help: 'Profit in R-multiples before stop moves to breakeven' },
      { key: 'be_offset', label: 'BE Offset ($)', type: 'float', default: 0.0, min: 0.0, max: 2.0, help: 'Small offset above entry for breakeven (covers commissions)' },
      { key: 'target_atr_mult', label: 'Target ATR\u00d7 (0=none)', type: 'float', default: 3.0, min: 0.0, max: 10.0, help: 'ATR target multiplier; 0 = exit via signal only' },
    ],
    stopSummary: 'ATR x{stop_atr_mult} -> BE at {be_activation_r}R',
    targetSummary: 'ATR x{target_atr_mult}',
    contextualHelp: [
      'Standard ATR stop that jumps to breakeven once the trade reaches the activation R threshold.',
      'Once at breakeven, the worst outcome is a scratch trade (plus any offset).',
      'The BE offset adds a small dollar amount above entry to cover commissions and slippage.',
      'Commonly paired with a 3x ATR target for a defined risk:reward profile.',
    ],
  },
};

/* ========================================================================
   Mock Pack Instances
   ======================================================================== */

const initialPacks: PackInstance[] = [
  {
    id: 'atr_default',
    templateKey: 'atr_based',
    versionName: 'Default',
    enabled: true,
    isDefault: true,
    paramValues: { stop_atr_mult: 1.5, target_atr_mult: 3.0 },
    description: '1.5x ATR stop, 3x ATR target',
  },
  {
    id: 'atr_tight',
    templateKey: 'atr_based',
    versionName: 'Tight',
    enabled: true,
    isDefault: true,
    paramValues: { stop_atr_mult: 1.0, target_atr_mult: 2.0 },
    description: '1x ATR stop, 2x ATR target',
  },
  {
    id: 'fixed_1_2',
    templateKey: 'fixed_dollar',
    versionName: '$1 / $2',
    enabled: true,
    isDefault: true,
    paramValues: { stop_amount: 1.0, target_amount: 2.0 },
    description: '$1 stop, $2 target',
  },
  {
    id: 'pct_half_one',
    templateKey: 'percentage',
    versionName: '0.5% / 1%',
    enabled: false,
    isDefault: true,
    paramValues: { stop_pct: 0.5, target_pct: 1.0 },
    description: '0.5% stop, 1% target',
  },
  {
    id: 'swing_2r',
    templateKey: 'swing',
    versionName: 'Swing 2R',
    enabled: true,
    isDefault: true,
    paramValues: { lookback: 5, padding: 0.05, rr_ratio: 2.0 },
    description: 'Swing-based stop with 2:1 reward',
  },
  {
    id: 'atr_trailing_default',
    templateKey: 'atr_trailing',
    versionName: 'Default',
    enabled: false,
    isDefault: true,
    paramValues: { stop_atr_mult: 1.5, trail_atr_mult: 1.0, trail_activation_r: 0.5, target_atr_mult: 0.0 },
    description: '1.5x ATR stop with 1x ATR trailing after 0.5R',
  },
  {
    id: 'breakeven_1r',
    templateKey: 'breakeven_stop',
    versionName: 'BE at 1R',
    enabled: false,
    isDefault: true,
    paramValues: { stop_atr_mult: 1.5, be_activation_r: 1.0, be_offset: 0.0, target_atr_mult: 3.0 },
    description: '1.5x ATR stop, moves to breakeven at 1R, 3x ATR target',
  },
];

/* ========================================================================
   Mock Preview Data
   ======================================================================== */

const mockStopPreviews: Record<string, StopPreviewEntry[]> = {
  atr_based: [
    { label: 'LONG entry (ATR=$1.20)', entryPrice: '$487.50', stopPrice: '$485.70', targetPrice: '$491.10', riskReward: '2.0R' },
    { label: 'SHORT entry (ATR=$1.20)', entryPrice: '$487.50', stopPrice: '$489.30', targetPrice: '$483.90', riskReward: '2.0R' },
    { label: 'LONG entry (ATR=$0.80)', entryPrice: '$487.50', stopPrice: '$486.30', targetPrice: '$489.90', riskReward: '2.0R' },
  ],
  fixed_dollar: [
    { label: 'LONG entry', entryPrice: '$487.50', stopPrice: '$486.50', targetPrice: '$489.50', riskReward: '2.0R' },
    { label: 'SHORT entry', entryPrice: '$487.50', stopPrice: '$488.50', targetPrice: '$485.50', riskReward: '2.0R' },
  ],
  percentage: [
    { label: 'LONG entry (0.5%/1%)', entryPrice: '$487.50', stopPrice: '$485.06', targetPrice: '$492.38', riskReward: '2.0R' },
    { label: 'SHORT entry (0.5%/1%)', entryPrice: '$487.50', stopPrice: '$489.94', targetPrice: '$482.63', riskReward: '2.0R' },
  ],
  swing: [
    { label: 'LONG (swing low $486.20)', entryPrice: '$487.50', stopPrice: '$486.15', targetPrice: '$490.20', riskReward: '2.0R' },
    { label: 'SHORT (swing high $488.90)', entryPrice: '$487.50', stopPrice: '$488.95', targetPrice: '$484.60', riskReward: '2.0R' },
  ],
  rr_ratio: [
    { label: 'ATR stop + 2R target', entryPrice: '$487.50', stopPrice: '$485.70', targetPrice: '$491.10', riskReward: '2.0R' },
  ],
  atr_trailing: [
    { label: 'Initial stop (1.5x ATR)', entryPrice: '$487.50', stopPrice: '$485.70', targetPrice: '--', riskReward: '--' },
    { label: 'After 0.5R profit', entryPrice: '$487.50', stopPrice: '$486.60', targetPrice: '--', riskReward: 'Trailing' },
    { label: 'After 1.0R profit', entryPrice: '$487.50', stopPrice: '$487.80', targetPrice: '--', riskReward: 'Trailing' },
  ],
  breakeven_stop: [
    { label: 'Initial stop (1.5x ATR)', entryPrice: '$487.50', stopPrice: '$485.70', targetPrice: '$491.10', riskReward: '2.0R' },
    { label: 'After 1R profit (BE)', entryPrice: '$487.50', stopPrice: '$487.50', targetPrice: '$491.10', riskReward: 'Free' },
  ],
};

/* ========================================================================
   Style Constants
   ======================================================================== */

const categoryColors: Record<string, { color: string; bg: string }> = {
  'Volatility': { color: 'var(--orange)', bg: 'var(--orange-muted)' },
  'Fixed': { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  'Structure': { color: 'var(--accent)', bg: 'var(--accent-muted)' },
  'Composite': { color: 'var(--purple)', bg: 'rgba(168,85,247,0.15)' },
  'Trailing': { color: 'var(--teal)', bg: 'rgba(20,184,166,0.15)' },
};

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
   Helpers
   ======================================================================== */

function getTemplate(pack: PackInstance): TemplateDef {
  return templates[pack.templateKey];
}

function getPackDisplayName(pack: PackInstance): string {
  const tpl = getTemplate(pack);
  return `${tpl.name} (${pack.versionName})`;
}

function formatStopSummary(pack: PackInstance): string {
  const tpl = getTemplate(pack);
  let summary = tpl.stopSummary;
  for (const [key, val] of Object.entries(pack.paramValues)) {
    summary = summary.replace(`{${key}}`, String(val));
  }
  return summary;
}

function formatTargetSummary(pack: PackInstance): string {
  const tpl = getTemplate(pack);
  let summary = tpl.targetSummary;
  for (const [key, val] of Object.entries(pack.paramValues)) {
    summary = summary.replace(`{${key}}`, String(val));
  }
  // Handle "0" target = no target
  const targetParams = tpl.parameters.filter((p) => p.key.includes('target'));
  if (targetParams.length > 0) {
    const val = pack.paramValues[targetParams[0].key];
    if (val === 0 || val === '0') return 'None';
  }
  return summary;
}

function formatParamSummary(pack: PackInstance): string {
  const tpl = getTemplate(pack);
  return tpl.parameters
    .filter((p) => {
      if (!p.visibleWhen) return true;
      return pack.paramValues[p.visibleWhen.field] === p.visibleWhen.value;
    })
    .map((p) => `${p.label}: ${pack.paramValues[p.key]}`)
    .join(', ');
}

function getCategoriesWithPacks(packs: PackInstance[]): { category: string; packs: PackInstance[] }[] {
  const categoryOrder = ['Volatility', 'Fixed', 'Structure', 'Composite', 'Trailing'];
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
   Sub-components
   ======================================================================== */

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

function StopTargetBadge({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div
      className="px-2.5 py-1.5 rounded-lg text-center"
      style={{ background: 'var(--bg-input)' }}
    >
      <span className="text-xs block" style={{ color: 'var(--text-muted)' }}>{label}</span>
      <span className="text-sm font-mono font-semibold" style={{ color }}>{value}</span>
    </div>
  );
}

/* ========================================================================
   Detail: Parameters Tab
   ======================================================================== */

function ParametersTab({ pack, tpl }: { pack: PackInstance; tpl: TemplateDef }) {
  const [stopMethod, setStopMethod] = useState(pack.paramValues.stop_method || 'atr');

  const visibleParams = tpl.parameters.filter((p) => {
    if (!p.visibleWhen) return true;
    const currentVal = p.visibleWhen.field === 'stop_method' ? stopMethod : pack.paramValues[p.visibleWhen.field];
    return currentVal === p.visibleWhen.value;
  });

  return (
    <div className="space-y-4">
      <Card>
        <div className="flex items-center justify-between mb-5">
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
            Pack Parameters
          </h3>
          <div className="flex items-center gap-2">
            <CategoryBadge category={tpl.category} />
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
              Template: {tpl.name}
            </span>
          </div>
        </div>

        <div className="space-y-4">
          {visibleParams.map((param) => (
            <div key={param.key}>
              <div className="flex items-center gap-4">
                <label className="text-sm w-52 flex-shrink-0" style={{ color: 'var(--text-secondary)' }}>
                  {param.label}
                </label>

                {param.type === 'select' ? (
                  <select
                    className="px-3 py-2 rounded-lg text-sm flex-1"
                    style={selectStyle}
                    defaultValue={pack.paramValues[param.key] as string}
                    onChange={(e) => {
                      if (param.key === 'stop_method') setStopMethod(e.target.value);
                    }}
                  >
                    {param.options?.map((opt) => (
                      <option key={opt} value={opt}>
                        {opt.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="number"
                    step={param.type === 'float' ? 0.1 : 1}
                    min={param.min}
                    max={param.max}
                    className="px-3 py-2 rounded-lg text-sm flex-1 font-mono"
                    style={inputStyle}
                    defaultValue={pack.paramValues[param.key] as number}
                  />
                )}
              </div>
              <div className="flex gap-4 mt-1">
                <span className="w-52 flex-shrink-0" />
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  {param.type === 'int' && `Integer | Range: ${param.min} - ${param.max} | Default: ${param.default}`}
                  {param.type === 'float' && `Decimal | Range: ${param.min} - ${param.max} | Default: ${param.default}`}
                  {param.type === 'select' && `Select | Options: ${param.options?.join(', ')}`}
                  {param.help ? ` -- ${param.help}` : ''}
                </span>
              </div>
            </div>
          ))}
        </div>

        {/* Stop / Target summary */}
        <div
          className="mt-5 flex items-center gap-4"
        >
          <StopTargetBadge label="Stop" value={formatStopSummary(pack)} color="var(--red)" />
          <span className="text-sm" style={{ color: 'var(--text-muted)' }}>&#8594;</span>
          <StopTargetBadge label="Target" value={formatTargetSummary(pack)} color="var(--green)" />
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

      {/* Contextual help */}
      <Card>
        <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>
          How {tpl.name} Works
        </h3>
        <div className="space-y-2">
          {tpl.contextualHelp.map((tip, i) => (
            <div key={i} className="flex gap-2">
              <span className="text-xs mt-0.5 flex-shrink-0" style={{ color: 'var(--accent)' }}>&#9679;</span>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                {tip}
              </p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

/* ========================================================================
   Detail: Outputs Tab
   ======================================================================== */

function OutputsTab({ pack, tpl }: { pack: PackInstance; tpl: TemplateDef }) {
  return (
    <div className="grid grid-cols-2 gap-6">
      {/* Stop Config */}
      <Card>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
            Stop-Loss Configuration
          </h3>
          <span
            className="text-xs px-2 py-0.5 rounded font-mono"
            style={{ background: 'var(--red-muted)', color: 'var(--red)' }}
          >
            STOP
          </span>
        </div>

        <div
          className="px-4 py-3 rounded-lg mb-4"
          style={{ background: 'var(--bg-input)' }}
        >
          <div className="text-lg font-mono font-semibold mb-1" style={{ color: 'var(--red)' }}>
            {formatStopSummary(pack)}
          </div>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {pack.description}
          </p>
        </div>

        <div className="space-y-2">
          <div className="text-xs font-medium mb-1" style={{ color: 'var(--text-muted)' }}>
            Stop Method Details
          </div>
          {pack.templateKey === 'atr_based' && (
            <>
              <InfoRow label="Method" value="ATR (Average True Range)" />
              <InfoRow label="ATR Period" value="14 bars (engine default)" />
              <InfoRow label="Multiplier" value={`${pack.paramValues.stop_atr_mult}x`} />
              <InfoRow label="Formula" value="entry +/- (ATR x multiplier)" mono />
            </>
          )}
          {pack.templateKey === 'fixed_dollar' && (
            <>
              <InfoRow label="Method" value="Fixed Dollar" />
              <InfoRow label="Distance" value={`$${pack.paramValues.stop_amount}`} />
              <InfoRow label="Formula" value="entry +/- dollar_amount" mono />
            </>
          )}
          {pack.templateKey === 'percentage' && (
            <>
              <InfoRow label="Method" value="Percentage" />
              <InfoRow label="Distance" value={`${pack.paramValues.stop_pct}%`} />
              <InfoRow label="Formula" value="entry x (1 +/- pct/100)" mono />
            </>
          )}
          {pack.templateKey === 'swing' && (
            <>
              <InfoRow label="Method" value="Swing Point" />
              <InfoRow label="Lookback" value={`${pack.paramValues.lookback} bars`} />
              <InfoRow label="Padding" value={`$${pack.paramValues.padding}`} />
              <InfoRow label="Formula" value="swing_low - padding (LONG)" mono />
            </>
          )}
          {pack.templateKey === 'atr_trailing' && (
            <>
              <InfoRow label="Initial" value={`ATR x ${pack.paramValues.stop_atr_mult}`} />
              <InfoRow label="Trail" value={`ATR x ${pack.paramValues.trail_atr_mult}`} />
              <InfoRow label="Activation" value={`${pack.paramValues.trail_activation_r}R profit`} />
            </>
          )}
          {pack.templateKey === 'breakeven_stop' && (
            <>
              <InfoRow label="Initial" value={`ATR x ${pack.paramValues.stop_atr_mult}`} />
              <InfoRow label="BE Trigger" value={`${pack.paramValues.be_activation_r}R profit`} />
              <InfoRow label="BE Offset" value={`$${pack.paramValues.be_offset}`} />
            </>
          )}
          {pack.templateKey === 'rr_ratio' && (
            <>
              <InfoRow label="Stop Method" value={String(pack.paramValues.stop_method).replace(/_/g, ' ')} />
              <InfoRow label="R:R Ratio" value={`${pack.paramValues.rr_ratio}R`} />
            </>
          )}
        </div>
      </Card>

      {/* Target Config */}
      <Card>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
            Take-Profit Configuration
          </h3>
          <span
            className="text-xs px-2 py-0.5 rounded font-mono"
            style={{ background: 'var(--green-muted)', color: 'var(--green)' }}
          >
            TARGET
          </span>
        </div>

        <div
          className="px-4 py-3 rounded-lg mb-4"
          style={{ background: 'var(--bg-input)' }}
        >
          <div className="text-lg font-mono font-semibold mb-1" style={{ color: 'var(--green)' }}>
            {formatTargetSummary(pack)}
          </div>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {formatTargetSummary(pack) === 'None'
              ? 'No target configured -- exits via stop loss, signal triggers, or bar count only.'
              : 'Profit target for automatic position exit.'}
          </p>
        </div>

        <div className="space-y-2">
          <div className="text-xs font-medium mb-1" style={{ color: 'var(--text-muted)' }}>
            Exit Priority Order
          </div>
          {[
            { label: '1. Stop Loss', desc: 'Protective exit at stop price', color: 'var(--red)' },
            { label: '2. Take Profit', desc: 'Target exit at profit price', color: 'var(--green)' },
            { label: '3. Signal Exit', desc: 'Opposite signal or trigger-based exit', color: 'var(--blue)' },
            { label: '4. Bar Count', desc: 'Time-based exit after N candles', color: 'var(--orange)' },
          ].map((item) => (
            <div
              key={item.label}
              className="flex items-center gap-2 px-3 py-2 rounded"
              style={{ background: 'var(--bg-input)' }}
            >
              <span className="text-xs font-semibold" style={{ color: item.color }}>
                {item.label}
              </span>
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                {item.desc}
              </span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

function InfoRow({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-center justify-between px-3 py-1.5 rounded" style={{ background: 'var(--bg-input)' }}>
      <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{label}</span>
      <span
        className={`text-xs font-semibold ${mono ? 'font-mono' : ''}`}
        style={{ color: 'var(--text-primary)' }}
      >
        {value}
      </span>
    </div>
  );
}

/* ========================================================================
   Detail: Preview Tab
   ======================================================================== */

function PreviewTab({ pack, tpl }: { pack: PackInstance; tpl: TemplateDef }) {
  const [symbol, setSymbol] = useState('NVDA');

  const stopPreviews = mockStopPreviews[pack.templateKey] || [];

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
          {['1Min', '5Min', '15Min', '1H'].map((tf) => (
            <option key={tf}>{tf}</option>
          ))}
        </select>
      </div>

      {/* Chart showing stop placement */}
      <Card>
        <ChartPlaceholder
          label={`${symbol} price chart with ${tpl.name} stop/target levels overlaid`}
          height={380}
        />
      </Card>

      {/* Stop Placement Examples */}
      <Card>
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
            Stop Placement Examples
          </h4>
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {stopPreviews.length} scenario{stopPreviews.length !== 1 ? 's' : ''}
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
              gridTemplateColumns: '2fr 1fr 1fr 1fr 1fr',
              gap: '8px',
            }}
          >
            <span>Scenario</span>
            <span>Entry</span>
            <span>Stop</span>
            <span>Target</span>
            <span>R:R</span>
          </div>
          {/* Rows */}
          {stopPreviews.map((row, i) => (
            <div
              key={i}
              className="grid text-xs px-3 py-2.5 border-t items-center"
              style={{
                borderColor: 'var(--border)',
                background: 'var(--bg-card)',
                gridTemplateColumns: '2fr 1fr 1fr 1fr 1fr',
                gap: '8px',
              }}
            >
              <span className="font-medium" style={{ color: 'var(--text-primary)' }}>
                {row.label}
              </span>
              <span className="font-mono" style={{ color: 'var(--text-primary)' }}>
                {row.entryPrice}
              </span>
              <span className="font-mono" style={{ color: 'var(--red)' }}>
                {row.stopPrice}
              </span>
              <span className="font-mono" style={{ color: 'var(--green)' }}>
                {row.targetPrice}
              </span>
              <span
                className="font-mono px-1.5 py-0.5 rounded text-center"
                style={{
                  background: row.riskReward === '--' ? 'var(--bg-input)' : 'var(--accent-muted)',
                  color: row.riskReward === '--' ? 'var(--text-muted)' : 'var(--accent)',
                }}
              >
                {row.riskReward}
              </span>
            </div>
          ))}
        </div>
      </Card>

      {/* Visual stop diagram for trailing/breakeven */}
      {(pack.templateKey === 'atr_trailing' || pack.templateKey === 'breakeven_stop') && (
        <Card>
          <h4 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>
            Stop Lifecycle
          </h4>
          <div className="flex items-center gap-2 flex-wrap">
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-full" style={{ background: 'var(--red)' }} />
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Initial Stop</span>
            </div>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>&#8594;</span>
            {pack.templateKey === 'atr_trailing' && (
              <>
                <div className="flex items-center gap-1.5">
                  <div className="w-3 h-3 rounded-full" style={{ background: 'var(--orange)' }} />
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    Trail activates at {String(pack.paramValues.trail_activation_r)}R
                  </span>
                </div>
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>&#8594;</span>
                <div className="flex items-center gap-1.5">
                  <div className="w-3 h-3 rounded-full" style={{ background: 'var(--green)' }} />
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Stop ratchets with price</span>
                </div>
              </>
            )}
            {pack.templateKey === 'breakeven_stop' && (
              <div className="flex items-center gap-1.5">
                <div className="w-3 h-3 rounded-full" style={{ background: 'var(--green)' }} />
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  Moves to breakeven at {String(pack.paramValues.be_activation_r)}R
                  {Number(pack.paramValues.be_offset) > 0 ? ` (+$${pack.paramValues.be_offset} offset)` : ''}
                </span>
              </div>
            )}
          </div>
        </Card>
      )}
    </div>
  );
}

/* ========================================================================
   Detail: Code Tab
   ======================================================================== */

function CodeTab({ pack, tpl }: { pack: PackInstance; tpl: TemplateDef }) {
  const [showStopBuilder, setShowStopBuilder] = useState(false);
  const [showTargetBuilder, setShowTargetBuilder] = useState(false);

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

  const visibleParams = tpl.parameters.filter((p) => {
    if (!p.visibleWhen) return true;
    return pack.paramValues[p.visibleWhen.field] === p.visibleWhen.value;
  });

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
            {visibleParams.map((p) => (
              <div key={p.key} className="flex justify-between">
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{p.label}</span>
                <span className="text-xs font-mono font-semibold" style={{ color: 'var(--text-primary)' }}>
                  {String(pack.paramValues[p.key])}
                </span>
              </div>
            ))}
          </div>
        </div>
        <div className="mt-3 flex gap-4">
          <div className="flex gap-2">
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Stop:</span>
            <span className="text-xs font-mono" style={{ color: 'var(--red)' }}>{formatStopSummary(pack)}</span>
          </div>
          <div className="flex gap-2">
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Target:</span>
            <span className="text-xs font-mono" style={{ color: 'var(--green)' }}>{formatTargetSummary(pack)}</span>
          </div>
        </div>
      </Card>

      {/* Stop Builder Function */}
      <Card>
        <button
          className="flex items-center justify-between w-full text-left"
          onClick={() => setShowStopBuilder(!showStopBuilder)}
        >
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
            Stop Builder Function
          </h3>
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {showStopBuilder ? 'Collapse' : 'Expand'}
          </span>
        </button>
        {showStopBuilder && (
          <div className="mt-3">
            <pre style={codeBlockStyle}>
{`def build_stop(params):
    """
    Build stop-loss config for ${tpl.name}.
    Category: ${tpl.category}

    Parameters:
${visibleParams.filter((p) => p.key.includes('stop') || p.key === 'lookback' || p.key === 'padding' || p.key === 'trail'  || p.key.includes('be_')).map((p) => `        ${p.key}: ${p.label} (${p.type}, range ${p.min}-${p.max})`).join('\n')}

    Returns dict: {"method": "...", ...params}
    """
    # Implementation in risk_management_packs.py
    ...`}
            </pre>
          </div>
        )}
      </Card>

      {/* Target Builder Function */}
      <Card>
        <button
          className="flex items-center justify-between w-full text-left"
          onClick={() => setShowTargetBuilder(!showTargetBuilder)}
        >
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
            Target Builder Function
          </h3>
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {showTargetBuilder ? 'Collapse' : 'Expand'}
          </span>
        </button>
        {showTargetBuilder && (
          <div className="mt-3">
            <pre style={codeBlockStyle}>
{`def build_target(params):
    """
    Build take-profit config for ${tpl.name}.

    Returns dict or None (no target).
    Target value of 0 = no target configured.
    """
    # Implementation in risk_management_packs.py
    ...`}
            </pre>
          </div>
        )}
      </Card>

      {/* Pack Config JSON */}
      <Card>
        <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>
          Pack Configuration (JSON)
        </h3>
        <pre style={codeBlockStyle}>
{JSON.stringify({
  id: pack.id,
  base_template: pack.templateKey,
  version: pack.versionName,
  description: pack.description,
  enabled: pack.enabled,
  is_default: pack.isDefault,
  parameters: pack.paramValues,
}, null, 2)}
        </pre>
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
            Reset all parameters to template defaults.
          </p>
          <button className="px-4 py-2 rounded-lg text-sm" style={btnSecondary}>
            Reset to Defaults
          </button>
        </div>

        {/* Delete */}
        <div className="pt-4 border-t" style={{ borderColor: 'var(--border)' }}>
          <p className="text-sm mb-2" style={{ color: 'var(--text-secondary)' }}>
            Delete this pack permanently. This cannot be undone.
            Strategies using this pack will fall back to the default ATR-Based pack.
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
              Default packs cannot be deleted. You can disable them instead.
            </p>
          )}
        </div>
      </div>
    </Card>
  );
}

/* ========================================================================
   Detail View (all 5 tabs)
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

      <TabBar tabs={['Parameters', 'Outputs', 'Preview', 'Code', 'Danger Zone']}>
        {(tab) => (
          <div>
            {tab === 'Parameters' && <ParametersTab pack={pack} tpl={tpl} />}
            {tab === 'Outputs' && <OutputsTab pack={pack} tpl={tpl} />}
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

          {/* Stop / Target summary */}
          <div className="flex items-center gap-3 mb-2">
            <span className="text-xs">
              <span style={{ color: 'var(--text-muted)' }}>Stop: </span>
              <span className="font-mono" style={{ color: 'var(--red)' }}>{formatStopSummary(pack)}</span>
            </span>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>|</span>
            <span className="text-xs">
              <span style={{ color: 'var(--text-muted)' }}>Target: </span>
              <span className="font-mono" style={{ color: 'var(--green)' }}>{formatTargetSummary(pack)}</span>
            </span>
          </div>

          {/* Parameter summary */}
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {formatParamSummary(pack)}
          </p>
        </div>

        {/* Right side: category + actions */}
        <div className="flex flex-col items-end gap-2 flex-shrink-0">
          <CategoryBadge category={tpl.category} />
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
    <Modal title="Create New Risk Management Pack" isOpen={isOpen} onClose={onClose} width="560px">
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
              {tpl.parameters.filter((p) => !p.visibleWhen).length} params
            </span>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs">
              <span style={{ color: 'var(--text-muted)' }}>Stop: </span>
              <span className="font-mono" style={{ color: 'var(--red)' }}>{tpl.stopSummary}</span>
            </span>
            <span className="text-xs">
              <span style={{ color: 'var(--text-muted)' }}>Target: </span>
              <span className="font-mono" style={{ color: 'var(--green)' }}>{tpl.targetSummary}</span>
            </span>
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
            placeholder="e.g. Tight, Aggressive, Conservative..."
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
            {tpl.parameters
              .filter((p) => !p.visibleWhen)
              .map((p) => `${p.label}: ${p.default}`)
              .join('  |  ')}
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

export default function RiskManagementV2() {
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
    const newId = `${pack.templateKey.replace(/_/g, '-')}-copy-${Date.now()}`;
    const copy: PackInstance = {
      ...pack,
      id: newId,
      versionName: `${pack.versionName} Copy`,
      isDefault: false,
      paramValues: { ...pack.paramValues },
    };
    setPacks((prev) => [...prev, copy]);
  }

  function handleCreatePack(templateKey: string, versionName: string) {
    const tpl = templates[templateKey];
    const newId = `${templateKey.replace(/_/g, '-')}-${versionName.toLowerCase().replace(/\s+/g, '-')}-${Date.now()}`;
    const paramValues: Record<string, number | string> = {};
    for (const p of tpl.parameters) paramValues[p.key] = p.default;

    const newPack: PackInstance = {
      id: newId,
      templateKey,
      versionName,
      enabled: true,
      isDefault: false,
      paramValues,
      description: `Custom ${tpl.name} configuration`,
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
        title="Risk Management Packs"
        subtitle="Stop-loss and take-profit configurations from shared parameters"
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
