'use client';

import { useState, useMemo, useEffect, useRef } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import Modal from '@/components/Modal';
import ChartPlaceholder from '@/components/ChartPlaceholder';
import Link from 'next/link';
import { useConfluenceGroups, useConfluenceTemplates } from '@/hooks/queries/usePacks';
import type { ConfluenceGroupDTO, ConfluenceTemplateDTO } from '@/hooks/queries/usePacks';
import { useSaveConfluenceGroups } from '@/hooks/mutations/usePackMutations';

/* ========================================================================
   Types
   ======================================================================== */

interface PackParam {
  key: string;
  label: string;
  type: 'int' | 'float';
  value: number;
  default: number;
  min: number;
  max: number;
}

interface PlotSetting {
  key: string;
  label: string;
  value: string;
}

interface OutputDef {
  code: string;
  description: string;
}

interface ExecConfig {
  enabled: boolean;
  referenceBar: 0 | -1;
  orderType: 'market' | 'limit';
  holdSeconds?: number;
  confirmBarOffset?: 0 | 1;
  bailAction?: 'exit_market' | 'exit_limit' | 'exit_limit_breakeven';
  limitPriceSource?: 'level' | 'entry' | 'custom_offset';
  limitDurationSeconds?: number;
}

interface PackTrigger {
  id: string;
  name: string;
  sentiment: 'bullish' | 'bearish' | 'neutral';
  description: string;
  execVariants: Record<'C' | 'L' | 'LC' | 'CC', ExecConfig>;
}

interface TfPack {
  id: string;
  templateKey: string;
  name: string;
  version: string;
  tags: string[];
  enabled: boolean;
  isDefault: boolean;
  isSaved: boolean;
  params: PackParam[];
  plotSettings: PlotSetting[];
  outputs: OutputDef[];
  triggers: PackTrigger[];
}

/* ========================================================================
   Template Definitions (mirrors confluence_groups.py TEMPLATES)
   ======================================================================== */

interface TemplateDef {
  name: string;
  tags: string[];
  description: string;
  parameters: Omit<PackParam, 'value'>[];
  plotSettings: PlotSetting[];
  outputs: OutputDef[];
  triggers: PackTrigger[];
}

// Default execution configs per type
const defaultExecVariants = (opts?: { disableCC?: boolean }): Record<'C' | 'L' | 'LC' | 'CC', ExecConfig> => ({
  C:  { enabled: true,  referenceBar: 0,  orderType: 'market' },
  L:  { enabled: true,  referenceBar: -1, orderType: 'market', holdSeconds: 0 },
  LC: { enabled: true,  referenceBar: -1, orderType: 'market', confirmBarOffset: 0, bailAction: 'exit_market' },
  CC: { enabled: !opts?.disableCC, referenceBar: 0, orderType: 'market', confirmBarOffset: 1, bailAction: 'exit_market' },
});

const TEMPLATES: Record<string, TemplateDef> = {
  ema_stack: {
    name: 'EMA Stack',
    tags: ['Moving Averages', 'Trend'],
    description: 'Three EMAs for trend direction and momentum',
    parameters: [
      { key: 'short_period', label: 'Short Period', type: 'int', default: 9, min: 1, max: 200 },
      { key: 'mid_period', label: 'Mid Period', type: 'int', default: 21, min: 1, max: 200 },
      { key: 'long_period', label: 'Long Period', type: 'int', default: 200, min: 1, max: 500 },
    ],
    plotSettings: [
      { key: 'short_color', label: 'Short EMA Color', value: '#22c55e' },
      { key: 'mid_color', label: 'Mid EMA Color', value: '#eab308' },
      { key: 'long_color', label: 'Long EMA Color', value: '#ef4444' },
    ],
    outputs: [
      { code: 'SML', description: 'Full Bull Stack \u2014 Short > Mid > Long' },
      { code: 'SLM', description: 'Short leading, Mid dipped below Long' },
      { code: 'MSL', description: 'Mid leading, Short pulling back' },
      { code: 'MLS', description: 'Mid leading, bearish Short' },
      { code: 'LSM', description: 'Long on top, transitional' },
      { code: 'LMS', description: 'Full Bear Stack \u2014 Long > Mid > Short' },
    ],
    triggers: [
      { id: 'cross_bull', name: 'Short > Mid Cross', sentiment: 'bullish', description: 'Short EMA crosses above Mid EMA', execVariants: defaultExecVariants() },
      { id: 'cross_bear', name: 'Short < Mid Cross', sentiment: 'bearish', description: 'Short EMA crosses below Mid EMA', execVariants: defaultExecVariants() },
      { id: 'mid_cross_bull', name: 'Mid > Long Cross', sentiment: 'bullish', description: 'Mid EMA crosses above Long EMA', execVariants: defaultExecVariants() },
      { id: 'mid_cross_bear', name: 'Mid < Long Cross', sentiment: 'bearish', description: 'Mid EMA crosses below Long EMA', execVariants: defaultExecVariants() },
    ],
  },
  macd_line: {
    name: 'MACD Line',
    tags: ['Momentum'],
    description: 'MACD line vs Signal line with zero-line context',
    parameters: [
      { key: 'fast_period', label: 'Fast Period', type: 'int', default: 12, min: 1, max: 100 },
      { key: 'slow_period', label: 'Slow Period', type: 'int', default: 26, min: 1, max: 200 },
      { key: 'signal_period', label: 'Signal Period', type: 'int', default: 9, min: 1, max: 50 },
    ],
    plotSettings: [
      { key: 'macd_color', label: 'MACD Line Color', value: '#2563eb' },
      { key: 'signal_color', label: 'Signal Line Color', value: '#f97316' },
    ],
    outputs: [
      { code: 'M>S+', description: 'MACD above signal, above zero (strong bull)' },
      { code: 'M>S-', description: 'MACD above signal, below zero (recovering)' },
      { code: 'M<S-', description: 'MACD below signal, below zero (strong bear)' },
      { code: 'M<S+', description: 'MACD below signal, above zero (weakening)' },
    ],
    triggers: [
      { id: 'cross_bull', name: 'Bullish Cross', sentiment: 'bullish', description: 'MACD line crosses above Signal line', execVariants: defaultExecVariants() },
      { id: 'cross_bear', name: 'Bearish Cross', sentiment: 'bearish', description: 'MACD line crosses below Signal line', execVariants: defaultExecVariants() },
      { id: 'zero_cross_up', name: 'Zero Line Cross Up', sentiment: 'bullish', description: 'MACD line crosses above zero', execVariants: defaultExecVariants() },
      { id: 'zero_cross_down', name: 'Zero Line Cross Down', sentiment: 'bearish', description: 'MACD line crosses below zero', execVariants: defaultExecVariants() },
    ],
  },
  macd_histogram: {
    name: 'MACD Histogram',
    tags: ['Momentum'],
    description: 'MACD histogram momentum shifts',
    parameters: [
      { key: 'fast_period', label: 'Fast Period', type: 'int', default: 12, min: 1, max: 100 },
      { key: 'slow_period', label: 'Slow Period', type: 'int', default: 26, min: 1, max: 200 },
      { key: 'signal_period', label: 'Signal Period', type: 'int', default: 9, min: 1, max: 50 },
    ],
    plotSettings: [
      { key: 'pos_color', label: 'Positive Bar Color', value: '#22c55e' },
      { key: 'neg_color', label: 'Negative Bar Color', value: '#ef4444' },
    ],
    outputs: [
      { code: 'H+up', description: 'Positive & increasing (accelerating bull)' },
      { code: 'H+dn', description: 'Positive & decreasing (decelerating bull)' },
      { code: 'H-dn', description: 'Negative & decreasing (accelerating bear)' },
      { code: 'H-up', description: 'Negative & increasing (decelerating bear)' },
    ],
    triggers: [
      { id: 'flip_pos', name: 'Histogram Flip Positive', sentiment: 'bullish', description: 'Histogram crosses from negative to positive', execVariants: defaultExecVariants() },
      { id: 'flip_neg', name: 'Histogram Flip Negative', sentiment: 'bearish', description: 'Histogram crosses from positive to negative', execVariants: defaultExecVariants() },
      { id: 'momentum_shift_up', name: 'Momentum Shift Up', sentiment: 'bullish', description: 'Histogram direction reverses upward', execVariants: defaultExecVariants() },
      { id: 'momentum_shift_down', name: 'Momentum Shift Down', sentiment: 'bearish', description: 'Histogram direction reverses downward', execVariants: defaultExecVariants() },
    ],
  },
  vwap: {
    name: 'VWAP',
    tags: ['Volume', 'Trend'],
    description: 'Volume Weighted Average Price with standard deviation bands (7-zone system)',
    parameters: [
      { key: 'sd1_mult', label: 'Inner Band (SD1)', type: 'float', default: 1.0, min: 0.5, max: 5.0 },
      { key: 'sd2_mult', label: 'Outer Band (SD2)', type: 'float', default: 2.0, min: 0.5, max: 5.0 },
    ],
    plotSettings: [
      { key: 'vwap_color', label: 'VWAP Line', value: '#8b5cf6' },
      { key: 'sd1_color', label: 'SD1 Band', value: '#c4b5fd' },
      { key: 'sd2_color', label: 'SD2 Band', value: '#ddd6fe' },
    ],
    outputs: [
      { code: '>+2\u03c3', description: 'Price above VWAP + 2xSD (extended high)' },
      { code: '>+1\u03c3', description: 'Price between +1\u03c3 and +2\u03c3' },
      { code: '>V', description: 'Price between VWAP and +1\u03c3' },
      { code: '@V', description: 'Price at VWAP (\u00b10.5\u03c3)' },
      { code: '<V', description: 'Price between VWAP and -1\u03c3' },
      { code: '<-1\u03c3', description: 'Price between -1\u03c3 and -2\u03c3' },
      { code: '<-2\u03c3', description: 'Price below VWAP - 2xSD (extended low)' },
    ],
    triggers: [
      { id: 'cross_above', name: 'Cross Above VWAP', sentiment: 'bullish', description: 'Price crosses above VWAP', execVariants: defaultExecVariants() },
      { id: 'cross_below', name: 'Cross Below VWAP', sentiment: 'bearish', description: 'Price crosses below VWAP', execVariants: defaultExecVariants() },
      { id: 'enter_upper_extreme', name: 'Enter Upper Extreme', sentiment: 'bearish', description: 'Price enters >+2\u03c3 zone', execVariants: defaultExecVariants() },
      { id: 'enter_lower_extreme', name: 'Enter Lower Extreme', sentiment: 'bullish', description: 'Price enters <-2\u03c3 zone', execVariants: defaultExecVariants() },
      { id: 'return_vwap', name: 'Return to VWAP Zone', sentiment: 'neutral', description: 'Price returns to @V zone', execVariants: defaultExecVariants() },
    ],
  },
  rvol: {
    name: 'Relative Volume',
    tags: ['Volume'],
    description: 'Current volume relative to historical average',
    parameters: [
      { key: 'sma_period', label: 'SMA Period', type: 'int', default: 20, min: 5, max: 100 },
      { key: 'high_threshold', label: 'High Threshold', type: 'float', default: 1.5, min: 1.0, max: 5.0 },
      { key: 'extreme_threshold', label: 'Extreme Threshold', type: 'float', default: 3.0, min: 2.0, max: 10.0 },
    ],
    plotSettings: [
      { key: 'bar_color', label: 'Volume Bar', value: '#64748b' },
      { key: 'high_color', label: 'High Volume', value: '#f59e0b' },
      { key: 'extreme_color', label: 'Extreme Volume', value: '#ef4444' },
    ],
    outputs: [
      { code: 'EXTREME', description: 'Volume > 300% of average' },
      { code: 'HIGH', description: 'Volume > 150% of average' },
      { code: 'NORMAL', description: 'Volume 75\u2013150% of average' },
      { code: 'LOW', description: 'Volume 50\u201375% of average' },
      { code: 'MINIMAL', description: 'Volume < 50% of average' },
    ],
    triggers: [
      { id: 'spike', name: 'Volume Spike', sentiment: 'neutral', description: 'Volume crosses above high threshold', execVariants: defaultExecVariants() },
      { id: 'extreme', name: 'Extreme Volume', sentiment: 'neutral', description: 'Volume crosses above extreme threshold', execVariants: defaultExecVariants() },
      { id: 'fade', name: 'Volume Fade', sentiment: 'neutral', description: 'Volume drops below high threshold', execVariants: defaultExecVariants() },
    ],
  },
  utbot_v2: {
    name: 'UT Bot Confirmed',
    tags: ['Trend', 'Volatility'],
    description: 'ATR trailing stop with trend confirmation',
    parameters: [
      { key: 'atr_period', label: 'ATR Period', type: 'int', default: 10, min: 1, max: 50 },
      { key: 'atr_multiplier', label: 'ATR Multiplier', type: 'float', default: 1.0, min: 0.1, max: 5.0 },
    ],
    plotSettings: [
      { key: 'trail_color', label: 'Trail Stop Color', value: '#f59e0b' },
    ],
    outputs: [
      { code: 'BULL', description: 'Price above trailing stop (bullish trend)' },
      { code: 'BEAR', description: 'Price below trailing stop (bearish trend)' },
    ],
    triggers: [
      { id: 'buy', name: 'Buy Signal', sentiment: 'bullish', description: 'Trail stop flips bullish', execVariants: defaultExecVariants() },
      { id: 'sell', name: 'Sell Signal', sentiment: 'bearish', description: 'Trail stop flips bearish', execVariants: defaultExecVariants() },
    ],
  },
  ema_price_position_v2: {
    name: 'EMA Price Position',
    tags: ['Moving Averages', 'Trend'],
    description: 'Price position relative to three EMAs (24-state system)',
    parameters: [
      { key: 'short_period', label: 'Short Period', type: 'int', default: 9, min: 1, max: 200 },
      { key: 'mid_period', label: 'Mid Period', type: 'int', default: 21, min: 1, max: 200 },
      { key: 'long_period', label: 'Long Period', type: 'int', default: 200, min: 1, max: 500 },
    ],
    plotSettings: [
      { key: 'short_color', label: 'Short EMA', value: '#22c55e' },
      { key: 'mid_color', label: 'Mid EMA', value: '#eab308' },
      { key: 'long_color', label: 'Long EMA', value: '#ef4444' },
    ],
    outputs: [
      { code: 'P>S>M>L', description: 'Price above all EMAs, full bull' },
      { code: 'S>P>M>L', description: 'Price between Short and Mid, bull stack' },
      { code: 'S>M>P>L', description: 'Price between Mid and Long, weakening' },
      { code: 'S>M>L>P', description: 'Price below all EMAs, bear in bull stack' },
      { code: 'L>M>S>P', description: 'Price below all EMAs, full bear stack' },
      { code: 'L>M>P>S', description: 'Price between Mid and Short, bear stack' },
    ],
    triggers: [
      { id: 'cross_short_up', name: 'Price Cross Short EMA Up', sentiment: 'bullish', description: 'Price crosses above Short EMA', execVariants: defaultExecVariants() },
      { id: 'cross_short_down', name: 'Price Cross Short EMA Down', sentiment: 'bearish', description: 'Price crosses below Short EMA', execVariants: defaultExecVariants() },
      { id: 'cross_mid_up', name: 'Price Cross Mid EMA Up', sentiment: 'bullish', description: 'Price crosses above Mid EMA', execVariants: defaultExecVariants() },
      { id: 'cross_mid_down', name: 'Price Cross Mid EMA Down', sentiment: 'bearish', description: 'Price crosses below Mid EMA', execVariants: defaultExecVariants() },
    ],
  },
  bar_count: {
    name: 'Bar Count Exit',
    tags: ['Exit'],
    description: 'Exit after a fixed number of bars',
    parameters: [
      { key: 'candle_count', label: 'Bar Count', type: 'int', default: 4, min: 1, max: 100 },
    ],
    plotSettings: [],
    outputs: [],
    triggers: [
      { id: 'exit', name: 'Bar Count Exit', sentiment: 'neutral', description: 'Exit position after N bars have elapsed', execVariants: defaultExecVariants({ disableCC: true }) },
    ],
  },
};

/* ========================================================================
   API → V5 Mapping
   ======================================================================== */

function mapApiGroupToTfPack(
  group: ConfluenceGroupDTO,
  apiTemplates?: Record<string, ConfluenceTemplateDTO>,
): TfPack {
  const templateKey = group.base_template;
  const localTemplate = TEMPLATES[templateKey];
  const apiTemplate = apiTemplates?.[templateKey];

  // Build params: merge API parameters with schema from template
  const params: PackParam[] = [];
  if (localTemplate) {
    for (const paramDef of localTemplate.parameters) {
      const apiVal = group.parameters?.[paramDef.key];
      params.push({
        ...paramDef,
        value: apiVal !== undefined ? Number(apiVal) : paramDef.default,
      });
    }
  } else if (apiTemplate?.parameters_schema) {
    for (const [key, schema] of Object.entries(apiTemplate.parameters_schema)) {
      const apiVal = group.parameters?.[key];
      params.push({
        key,
        label: schema.label || key,
        type: (schema.type === 'float' ? 'float' : 'int') as 'int' | 'float',
        value: apiVal !== undefined ? Number(apiVal) : Number(schema.default ?? 0),
        default: Number(schema.default ?? 0),
        min: schema.min ?? 0,
        max: schema.max ?? 999,
      });
    }
  }

  // Plot settings: prefer local template, overlay API colors
  const plotSettings: PlotSetting[] = localTemplate
    ? localTemplate.plotSettings.map((ps) => ({
        ...ps,
        value: group.plot_settings?.colors?.[ps.key] || ps.value,
      }))
    : Object.entries(group.plot_settings?.colors || {}).map(([key, value]) => ({
        key,
        label: key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
        value,
      }));

  // Outputs & triggers from local template (display-only), fallback to API template
  const outputs: OutputDef[] = localTemplate?.outputs
    ?? (apiTemplate?.outputs
      ? Object.entries(apiTemplate.outputs).map(([code, description]) => ({ code, description }))
      : []);

  const triggers: PackTrigger[] = localTemplate?.triggers
    ?? (apiTemplate?.triggers || []).map((t) => ({
      id: t.id,
      name: t.name,
      sentiment: (t.direction === 'LONG' ? 'bullish' : t.direction === 'SHORT' ? 'bearish' : 'neutral') as 'bullish' | 'bearish' | 'neutral',
      description: t.name,
      execVariants: defaultExecVariants(),
    }));

  return {
    id: group.id,
    templateKey,
    name: localTemplate?.name || apiTemplate?.name || templateKey,
    version: group.version || 'Default',
    tags: localTemplate?.tags || (apiTemplate?.category ? [apiTemplate.category] : []),
    enabled: group.enabled,
    isDefault: group.is_default,
    isSaved: true,
    params,
    plotSettings,
    outputs,
    triggers,
  };
}

/* ========================================================================
   Style Constants
   ======================================================================== */

const tagColors: Record<string, { color: string; bg: string }> = {
  'Moving Averages': { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  Momentum: { color: 'var(--orange)', bg: 'var(--orange-muted)' },
  Volume: { color: 'var(--accent)', bg: 'var(--accent-muted)' },
  Volatility: { color: 'var(--red)', bg: 'var(--red-muted)' },
  Trend: { color: 'var(--green)', bg: 'var(--green-muted)' },
  Exit: { color: 'var(--text-muted)', bg: 'var(--bg-input)' },
};

// Display settings V5 spec: single color for all exec types, single color for all fidelity types
const EXEC_BADGE_COLOR = '#2196F3';
const FIDELITY_BADGE_COLOR = '#26C6DA';

const execLabels: Record<string, string> = {
  C: 'Close', L: 'Level', LC: 'Level-Close', CC: 'Close-Close',
};


/* ========================================================================
   Badge Components
   ======================================================================== */

function ExecBadge({ exec }: { exec: string }) {
  return (
    <span
      className="text-xs font-mono font-medium px-2 py-0.5 rounded-full"
      style={{ color: EXEC_BADGE_COLOR, background: EXEC_BADGE_COLOR + '20' }}
      title={execLabels[exec] || exec}
    >
      [{exec}]
    </span>
  );
}

function SentimentBadge({ sentiment }: { sentiment: string }) {
  const styles: Record<string, { color: string; bg: string; icon: string }> = {
    bullish: { color: 'var(--green)', bg: 'var(--green-muted)', icon: '\u2191' },
    bearish: { color: 'var(--red)', bg: 'var(--red-muted)', icon: '\u2193' },
    neutral: { color: 'var(--text-muted)', bg: 'var(--bg-input)', icon: '\u2194' },
  };
  const s = styles[sentiment] || styles.neutral;
  return (
    <span className="text-[10px] font-medium px-1.5 py-0.5 rounded" style={{ color: s.color, background: s.bg }}>
      {s.icon} {sentiment}
    </span>
  );
}

function FidelityBadge({ type }: { type: 'PB' | 'CB' }) {
  return (
    <span
      className="text-xs font-mono font-medium px-2 py-0.5 rounded-full"
      style={{ color: FIDELITY_BADGE_COLOR, background: FIDELITY_BADGE_COLOR + '20' }}
    >
      [{type}]
    </span>
  );
}

function TagBadge({ tag }: { tag: string }) {
  const style = tagColors[tag] || { color: 'var(--text-muted)', bg: 'var(--bg-input)' };
  return (
    <span className="text-[10px] font-medium px-1.5 py-0.5 rounded" style={{ color: style.color, background: style.bg }}>
      {tag}
    </span>
  );
}

function Toggle({ enabled, onChange }: { enabled: boolean; onChange: () => void }) {
  return (
    <button
      onClick={(e) => { e.stopPropagation(); onChange(); }}
      className="w-9 h-5 rounded-full relative flex-shrink-0 transition-colors"
      style={{
        background: enabled ? 'var(--accent)' : 'var(--bg-input)',
        border: enabled ? 'none' : '1px solid var(--border)',
      }}
    >
      <div
        className="w-3.5 h-3.5 rounded-full absolute transition-all"
        style={{
          background: enabled ? 'white' : 'var(--text-muted)',
          top: '3px',
          left: enabled ? '19px' : '3px',
        }}
      />
    </button>
  );
}

/* ========================================================================
   Detail View — Parameters Tab
   ======================================================================== */

function ParametersTab({ pack }: { pack: TfPack }) {
  const template = TEMPLATES[pack.templateKey];
  return (
    <Card>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
          Pack Parameters
        </h3>
        {pack.isSaved && (
          <span className="text-xs px-2 py-1 rounded-lg flex items-center gap-1.5" style={{ background: 'var(--orange-muted)', color: 'var(--orange)' }}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <rect x="3" y="11" width="18" height="11" rx="2" />
              <path d="M7 11V7a5 5 0 0110 0v4" />
            </svg>
            Parameters locked after save
          </span>
        )}
      </div>

      {pack.isSaved && (
        <div className="mb-4 px-3 py-2 rounded-lg text-xs" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>
          Parameters are locked to protect active strategies and portfolios. To use different parameters, create a new variation via the Copy button.
        </div>
      )}

      <div className="space-y-3">
        {pack.params.map((param) => (
          <div key={param.key} className="flex items-center gap-4">
            <label className="text-sm w-40 flex-shrink-0" style={{ color: 'var(--text-secondary)' }}>{param.label}</label>
            <input
              type="number"
              className="w-24 px-3 py-2 rounded-lg text-sm text-center font-mono"
              style={{
                background: 'var(--bg-input)',
                border: '1px solid var(--border)',
                color: pack.isSaved ? 'var(--text-muted)' : 'var(--text-primary)',
                cursor: pack.isSaved ? 'not-allowed' : 'text',
              }}
              value={param.value}
              disabled={pack.isSaved}
              readOnly={pack.isSaved}
            />
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
              {param.type === 'int' ? 'Integer' : 'Decimal'} ({param.min}–{param.max})
            </span>
            {param.value !== param.default && (
              <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ color: 'var(--orange)', background: 'var(--orange-muted)' }}>
                default: {param.default}
              </span>
            )}
          </div>
        ))}
      </div>

      {!pack.isSaved && (
        <div className="mt-6">
          <div className="px-3 py-2 rounded-lg text-xs flex items-center gap-2" style={{ background: 'var(--orange-muted)', color: 'var(--orange)' }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
              <line x1="12" y1="9" x2="12" y2="13" />
              <line x1="12" y1="17" x2="12.01" y2="17" />
            </svg>
            Once saved, parameters cannot be changed. Review all tabs and use the &quot;Save as Variation&quot; button in the header when ready.
          </div>
        </div>
      )}

      {template && (
        <div className="mt-6 pt-4 border-t" style={{ borderColor: 'var(--border)' }}>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            Template: <span className="font-medium" style={{ color: 'var(--text-secondary)' }}>{template.name}</span>
            {' \u00b7 '}
            {template.description}
          </p>
        </div>
      )}
    </Card>
  );
}

/* ========================================================================
   Detail View — Plot Settings Tab
   ======================================================================== */

function PlotSettingsTab({ pack }: { pack: TfPack }) {
  return (
    <Card>
      <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Visual Settings</h3>
      <div className="space-y-4">
        {pack.plotSettings.map((setting) => (
          <div key={setting.key} className="flex items-center gap-4">
            <label className="text-sm w-40 flex-shrink-0" style={{ color: 'var(--text-secondary)' }}>{setting.label}</label>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded border" style={{ background: setting.value, borderColor: 'var(--border)' }} />
              <input
                className="px-3 py-2 rounded-lg text-sm font-mono w-28"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                defaultValue={setting.value}
              />
            </div>
          </div>
        ))}
      </div>
      <button className="mt-6 px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--accent)', color: 'white' }}>
        Save Plot Settings
      </button>
    </Card>
  );
}

/* ========================================================================
   Detail View — Outputs & Triggers Tab
   ======================================================================== */

function ExecSettingsRow({ label, config, locked }: { label: string; config: ExecConfig; locked: boolean }) {
  if (!config.enabled) return (
    <div className="flex items-center gap-2 px-2 py-1.5 rounded" style={{ background: 'var(--bg-card)', opacity: 0.4 }}>
      <ExecBadge exec={label} />
      <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Not available</span>
    </div>
  );

  return (
    <div className="px-2 py-1.5 rounded" style={{ background: 'var(--bg-card)' }}>
      <div className="flex items-center gap-2 mb-1">
        <ExecBadge exec={label} />
        <span className="text-[10px]" style={{ color: 'var(--text-secondary)' }}>{execLabels[label]}</span>
        {locked && (
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" style={{ color: 'var(--text-muted)' }}>
            <rect x="3" y="11" width="18" height="11" rx="2" /><path d="M7 11V7a5 5 0 0110 0v4" />
          </svg>
        )}
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-0.5 ml-6">
        <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
          ref: <span className="font-mono">{config.referenceBar === -1 ? 'prev bar' : 'current bar'}</span>
        </span>
        <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
          order: <span className="font-mono">{config.orderType}</span>
        </span>
        {config.holdSeconds !== undefined && (
          <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
            hold: <span className="font-mono">{config.holdSeconds}s</span>
          </span>
        )}
        {config.confirmBarOffset !== undefined && (
          <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
            confirm: <span className="font-mono">+{config.confirmBarOffset} bar</span>
          </span>
        )}
        {config.bailAction && (
          <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
            bail: <span className="font-mono">{
              config.bailAction === 'exit_limit_breakeven' ? 'limit @ breakeven' :
              config.bailAction === 'exit_limit' ? 'limit exit' : 'market exit'
            }</span>
          </span>
        )}
      </div>
    </div>
  );
}

function OutputsTriggersTab({ pack }: { pack: TfPack }) {
  const [expandedTrigger, setExpandedTrigger] = useState<string | null>(null);
  const totalVariants = pack.triggers.reduce((acc, t) => acc + (['C', 'L', 'LC', 'CC'] as const).filter((k) => t.execVariants[k].enabled).length, 0);

  return (
    <div className="space-y-6">
      {/* Outputs */}
      <Card>
        <div className="flex items-center gap-3 mb-4">
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Interpreter Outputs</h3>
          <FidelityBadge type="PB" />
          <FidelityBadge type="CB" />
          <span className="flex-1" />
          <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>
            {pack.outputs.length} states
          </span>
        </div>
        <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
          Each output state is available as both Previous Bar [PB] and Current Bar [CB] fidelity variants for cross-timeframe confluence.
        </p>
        <div className="space-y-2">
          {pack.outputs.map((output) => (
            <div key={output.code} className="flex items-start gap-3 px-3 py-2.5 rounded-lg" style={{ background: 'var(--bg-input)' }}>
              <span className="text-sm font-mono font-bold flex-shrink-0 mt-0.5" style={{ color: 'var(--text-primary)', minWidth: 56 }}>
                {output.code}
              </span>
              <span className="text-xs leading-relaxed" style={{ color: 'var(--text-muted)' }}>{output.description}</span>
            </div>
          ))}
          {pack.outputs.length === 0 && (
            <p className="text-xs px-3 py-2" style={{ color: 'var(--text-muted)' }}>No interpreter outputs (exit-only template)</p>
          )}
        </div>
      </Card>

      {/* Triggers */}
      <Card>
        <div className="flex items-center gap-3 mb-4">
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Available Triggers</h3>
          <ExecBadge exec="C" />
          <ExecBadge exec="L" />
          <ExecBadge exec="LC" />
          <ExecBadge exec="CC" />
          <span className="flex-1" />
          <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>
            {pack.triggers.length} base, {totalVariants} total
          </span>
        </div>
        <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
          Each trigger has up to 4 execution variants. Execution type settings are configured in the Trigger Parameters tab and apply uniformly to all triggers.
        </p>
        <div className="space-y-2">
          {pack.triggers.map((trigger) => {
            const isExpanded = expandedTrigger === trigger.id;
            const enabledVariants = (['C', 'L', 'LC', 'CC'] as const).filter((k) => trigger.execVariants[k].enabled);

            return (
              <div key={trigger.id} className="rounded-lg overflow-hidden" style={{ background: 'var(--bg-input)' }}>
                <button
                  className="w-full text-left px-3 py-2.5 flex items-center gap-2"
                  onClick={() => setExpandedTrigger(isExpanded ? null : trigger.id)}
                >
                  <span
                    className="text-[10px] flex-shrink-0 transition-transform"
                    style={{ color: 'var(--text-muted)', transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)' }}
                  >
                    {'\u25B6'}
                  </span>
                  <span className="text-sm font-medium flex-1" style={{ color: 'var(--text-primary)' }}>{trigger.name}</span>
                  <SentimentBadge sentiment={trigger.sentiment} />
                  <div className="flex gap-1">
                    {enabledVariants.map((v) => <ExecBadge key={v} exec={v} />)}
                  </div>
                </button>

                {isExpanded && (
                  <div className="px-3 pb-3 border-t" style={{ borderColor: 'var(--border)' }}>
                    <p className="text-xs my-2" style={{ color: 'var(--text-muted)' }}>{trigger.description}</p>

                    {/* Execution variant IDs */}
                    <div className="space-y-1">
                      {enabledVariants.map((v) => (
                        <div key={v} className="flex items-center gap-2 text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>
                          <ExecBadge exec={v} />
                          <span>{pack.id}_{trigger.id}_{v.toLowerCase()}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </Card>
    </div>
  );
}

/* ========================================================================
   Mock Preview Data
   ======================================================================== */

interface StateTimelineEntry { time: string; state: string; prevState: string }
interface TriggerEvent { time: string; trigger: string; direction: string; type: string; price: string; execType: string }

// Preview data: not wired to API yet — empty until live preview endpoint exists
const mockStateTimelines: Record<string, StateTimelineEntry[]> = {};
const mockTriggerEvents: Record<string, TriggerEvent[]> = {};

/* ========================================================================
   Detail View — Preview Tab
   ======================================================================== */

function PreviewTab({ pack }: { pack: TfPack }) {
  const [showConditions, setShowConditions] = useState(true);
  const [showTriggers, setShowTriggers] = useState(true);
  const [symbol, setSymbol] = useState('NVDA');

  const stateTimeline = mockStateTimelines[pack.templateKey] || [];
  const triggerEvents = mockTriggerEvents[pack.templateKey] || [];

  const selectStyle = { background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' };

  return (
    <div className="space-y-4">
      {/* Controls bar */}
      <div className="flex items-center gap-3 flex-wrap">
        <select className="px-3 py-2 rounded-lg text-sm" style={selectStyle} value={symbol} onChange={(e) => setSymbol(e.target.value)}>
          {['NVDA', 'SPY', 'AAPL', 'TSLA', 'MSFT', 'AMD', 'BTC/USD'].map((s) => <option key={s}>{s}</option>)}
        </select>
        <select className="px-3 py-2 rounded-lg text-sm" style={selectStyle}>
          {['1Min', '5Min', '15Min', '1H', '4H', '1D'].map((tf) => <option key={tf}>{tf}</option>)}
        </select>
        <select className="px-3 py-2 rounded-lg text-sm" style={selectStyle}>
          <option>RTH Only</option>
          <option>Extended Hours</option>
          <option>24/7</option>
        </select>

        <div className="border-l h-6 mx-1" style={{ borderColor: 'var(--border)' }} />

        <label className="flex items-center gap-1.5 cursor-pointer">
          <input type="checkbox" checked={showConditions} onChange={(e) => setShowConditions(e.target.checked)} className="w-3.5 h-3.5 rounded" />
          <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>Show Conditions</span>
        </label>
        <label className="flex items-center gap-1.5 cursor-pointer">
          <input type="checkbox" checked={showTriggers} onChange={(e) => setShowTriggers(e.target.checked)} className="w-3.5 h-3.5 rounded" />
          <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>Show Triggers</span>
        </label>
      </div>

      {/* Chart */}
      <Card>
        <ChartPlaceholder
          label={`${symbol} price chart with ${pack.name} (${pack.version}) overlay${showConditions ? ' + conditions' : ''}${showTriggers ? ' + triggers' : ''}`}
          height={380}
        />
      </Card>

      {/* Data tables */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Interpreter States */}
        <Card>
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Interpreter States</h4>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{stateTimeline.length} transitions</span>
          </div>
          <div className="rounded-lg overflow-hidden border" style={{ borderColor: 'var(--border)' }}>
            <div className="grid grid-cols-3 text-xs font-medium px-3 py-2" style={{ background: 'var(--bg-secondary)', color: 'var(--text-muted)' }}>
              <span>Time</span><span>State</span><span>Previous</span>
            </div>
            {stateTimeline.map((row, i) => {
              const changed = row.state !== row.prevState;
              return (
                <div key={i} className="grid grid-cols-3 text-xs px-3 py-2 border-t" style={{ borderColor: 'var(--border)', background: changed ? 'var(--accent-muted)' : 'var(--bg-card)' }}>
                  <span className="font-mono" style={{ color: 'var(--text-muted)' }}>{row.time}</span>
                  <span className="font-mono font-semibold" style={{ color: changed ? 'var(--accent)' : 'var(--text-primary)' }}>{row.state}</span>
                  <span className="font-mono" style={{ color: 'var(--text-muted)' }}>{row.prevState}</span>
                </div>
              );
            })}
            {stateTimeline.length === 0 && (
              <div className="px-3 py-4 text-xs text-center" style={{ color: 'var(--text-muted)' }}>No state data for this template</div>
            )}
          </div>
        </Card>

        {/* Trigger Events */}
        <Card>
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Trigger Events</h4>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{triggerEvents.length} events</span>
          </div>
          <div className="rounded-lg overflow-hidden border" style={{ borderColor: 'var(--border)' }}>
            <div className="grid text-xs font-medium px-3 py-2" style={{ background: 'var(--bg-secondary)', color: 'var(--text-muted)', gridTemplateColumns: '70px 1fr 70px 70px 35px', gap: '4px' }}>
              <span>Time</span><span>Trigger</span><span>Sentiment</span><span>Price</span><span>Exec</span>
            </div>
            {triggerEvents.map((row, i) => {
              const sentiment = row.direction === 'LONG' ? 'bullish' : row.direction === 'SHORT' ? 'bearish' : 'neutral';
              return (
                <div key={i} className="grid text-xs px-3 py-2 border-t items-center" style={{ borderColor: 'var(--border)', background: 'var(--bg-card)', gridTemplateColumns: '70px 1fr 70px 70px 35px', gap: '4px' }}>
                  <span className="font-mono" style={{ color: 'var(--text-muted)' }}>{row.time}</span>
                  <span className="font-medium truncate" style={{ color: 'var(--text-primary)' }}>{row.trigger}</span>
                  <SentimentBadge sentiment={sentiment} />
                  <span className="font-mono" style={{ color: 'var(--text-primary)' }}>{row.price}</span>
                  <ExecBadge exec={row.execType} />
                </div>
              );
            })}
            {triggerEvents.length === 0 && (
              <div className="px-3 py-4 text-xs text-center" style={{ color: 'var(--text-muted)' }}>No trigger data for this template</div>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}

/* ========================================================================
   Detail View — Code Tab
   ======================================================================== */

function CodeTab({ pack }: { pack: TfPack }) {
  const [showIndicatorCode, setShowIndicatorCode] = useState(false);
  const [showInterpreterCode, setShowInterpreterCode] = useState(false);
  const [showPineScript, setShowPineScript] = useState(false);

  const template = TEMPLATES[pack.templateKey];
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
        <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Active Parameters</h3>
        <div className="rounded-lg px-4 py-3" style={{ background: 'var(--bg-primary)', border: '1px solid var(--border)' }}>
          <div className="grid grid-cols-2 gap-y-2 gap-x-8">
            {pack.params.map((p) => (
              <div key={p.key} className="flex justify-between">
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{p.label}</span>
                <span className="text-xs font-mono font-semibold" style={{ color: 'var(--text-primary)' }}>{p.value}</span>
              </div>
            ))}
          </div>
        </div>
      </Card>

      {/* Indicator Function */}
      <Card>
        <button className="flex items-center justify-between w-full text-left" onClick={() => setShowIndicatorCode(!showIndicatorCode)}>
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Indicator Function</h3>
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{showIndicatorCode ? 'Collapse' : 'Expand'}</span>
        </button>
        {showIndicatorCode && (
          <div className="mt-3">
            <pre style={codeBlockStyle}>
{`def compute_${pack.templateKey}(df, ${pack.params.map((p) => `${p.key}=${p.value}`).join(', ')}):
    """
    Compute ${pack.name} indicator values.
    Template: ${pack.name} (${pack.tags.join(', ')})

    Parameters:
${pack.params.map((p) => `        ${p.key}: ${p.label} (${p.type}, range ${p.min}-${p.max})`).join('\n')}

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
        <button className="flex items-center justify-between w-full text-left" onClick={() => setShowInterpreterCode(!showInterpreterCode)}>
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Interpreter Function</h3>
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{showInterpreterCode ? 'Collapse' : 'Expand'}</span>
        </button>
        {showInterpreterCode && (
          <div className="mt-3">
            <pre style={codeBlockStyle}>
{`def interpret_${pack.templateKey}(row):
    """
    Classify bar state into one of ${pack.outputs.length} output categories.

    Outputs:
${pack.outputs.map((o) => `        ${o.code}: ${o.description}`).join('\n')}

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
        <button className="flex items-center justify-between w-full text-left" onClick={() => setShowPineScript(!showPineScript)}>
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Pine Script Reference</h3>
          <span className="text-xs px-2 py-0.5 rounded" style={{ color: 'var(--text-muted)', background: 'var(--bg-input)' }}>Optional</span>
        </button>
        {showPineScript && (
          <div className="mt-3">
            <pre style={codeBlockStyle}>
{`//@version=5
// ${pack.name} -- Reference implementation
// Parameters: ${pack.params.map((p) => `${p.key}=${p.value}`).join(', ')}
indicator("${pack.name}", overlay=true)

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
   Detail View — Danger Zone Tab
   ======================================================================== */

function DangerZoneTab({ pack }: { pack: TfPack }) {
  return (
    <Card>
      <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--red)' }}>Danger Zone</h3>
      <div className="space-y-6">
        <div>
          <label className="text-sm mb-2 block" style={{ color: 'var(--text-secondary)' }}>Rename Variation</label>
          <div className="flex gap-3">
            <input
              className="px-3 py-2 rounded-lg text-sm flex-1"
              style={{
                background: 'var(--bg-input)',
                border: '1px solid var(--border)',
                color: pack.isDefault ? 'var(--text-muted)' : 'var(--text-primary)',
                cursor: pack.isDefault ? 'not-allowed' : 'text',
              }}
              defaultValue={pack.version}
              disabled={pack.isDefault}
            />
            <button
              className="px-4 py-2 rounded-lg text-sm"
              style={{
                background: pack.isDefault ? 'var(--bg-input)' : 'var(--bg-card)',
                border: '1px solid var(--border)',
                color: pack.isDefault ? 'var(--text-muted)' : 'var(--text-secondary)',
                cursor: pack.isDefault ? 'not-allowed' : 'pointer',
              }}
              disabled={pack.isDefault}
            >
              Rename
            </button>
          </div>
          {pack.isDefault && <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>Default packs cannot be renamed.</p>}
        </div>
        <div className="pt-4 border-t" style={{ borderColor: 'var(--border)' }}>
          <p className="text-sm mb-2" style={{ color: 'var(--text-secondary)' }}>Delete this variation permanently.</p>
          <button
            className="px-4 py-2 rounded-lg text-sm font-medium"
            style={{
              background: pack.isDefault ? 'var(--bg-input)' : 'var(--red)',
              color: pack.isDefault ? 'var(--text-muted)' : 'white',
              cursor: pack.isDefault ? 'not-allowed' : 'pointer',
            }}
            disabled={pack.isDefault}
          >
            Delete Variation
          </button>
          {pack.isDefault && <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>Default packs cannot be deleted.</p>}
        </div>
      </div>
    </Card>
  );
}

/* ========================================================================
   Copy / New Variation Modal
   ======================================================================== */

function CopyVariationModal({ sourcePack, isOpen, onClose }: { sourcePack: TfPack; isOpen: boolean; onClose: () => void }) {
  const [versionName, setVersionName] = useState('');
  const [params, setParams] = useState<PackParam[]>(sourcePack.params.map((p) => ({ ...p })));

  function updateParam(key: string, value: number) {
    setParams((prev) => prev.map((p) => (p.key === key ? { ...p, value } : p)));
  }

  function resetParams() {
    setParams(sourcePack.params.map((p) => ({ ...p, value: p.default })));
  }

  const packIdPreview = `${sourcePack.templateKey}_${versionName ? versionName.toLowerCase().replace(/\s+/g, '_') : 'custom'}`;

  return (
    <Modal title={`Create Variation of ${sourcePack.name}`} isOpen={isOpen} onClose={onClose} width="600px">
      <div className="space-y-5">
        {/* Source info */}
        <div className="px-3 py-2.5 rounded-lg" style={{ background: 'var(--bg-input)' }}>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            Creating a new variation of <span className="font-medium" style={{ color: 'var(--text-secondary)' }}>{sourcePack.name}</span>.
            This will share the same outputs and triggers, but with your custom parameters.
          </p>
        </div>

        {/* Version name */}
        <div>
          <label className="text-sm mb-1.5 block" style={{ color: 'var(--text-secondary)' }}>Variation Name</label>
          <input
            className="w-full px-3 py-2 rounded-lg text-sm"
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
            placeholder="e.g. Scalping, Swing, Conservative..."
            value={versionName}
            onChange={(e) => setVersionName(e.target.value)}
          />
          <p className="text-[10px] font-mono mt-1" style={{ color: 'var(--text-muted)' }}>ID: {packIdPreview}</p>
        </div>

        {/* Parameters */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-sm" style={{ color: 'var(--text-secondary)' }}>Parameters</label>
            <button
              onClick={resetParams}
              className="text-[10px] px-2 py-0.5 rounded"
              style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-muted)' }}
            >
              Reset to Defaults
            </button>
          </div>
          <div className="space-y-2.5">
            {params.map((param) => (
              <div key={param.key} className="flex items-center gap-3">
                <label className="text-sm w-36 flex-shrink-0" style={{ color: 'var(--text-secondary)' }}>{param.label}</label>
                <input
                  type="number"
                  className="w-24 px-3 py-2 rounded-lg text-sm text-center font-mono"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                  value={param.value}
                  min={param.min}
                  max={param.max}
                  step={param.type === 'float' ? 0.1 : 1}
                  onChange={(e) => updateParam(param.key, parseFloat(e.target.value) || param.min)}
                />
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>({param.min}–{param.max})</span>
                {param.value !== param.default && (
                  <span className="text-[10px]" style={{ color: 'var(--orange)' }}>changed</span>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Lock warning */}
        <div className="px-3 py-2.5 rounded-lg flex items-start gap-2" style={{ background: 'var(--orange-muted)', color: 'var(--orange)' }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="flex-shrink-0 mt-0.5">
            <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
            <line x1="12" y1="9" x2="12" y2="13" />
            <line x1="12" y1="17" x2="12.01" y2="17" />
          </svg>
          <p className="text-xs leading-relaxed">
            Parameters will be <strong>locked permanently</strong> after saving. This protects strategies and portfolios using this variation. To try different settings, create another variation.
          </p>
        </div>

        {/* Actions */}
        <div className="flex gap-3 pt-1">
          <button
            className="px-4 py-2 rounded-lg text-sm font-medium flex-1"
            style={{
              background: versionName ? 'var(--accent)' : 'var(--bg-input)',
              color: versionName ? 'white' : 'var(--text-muted)',
              cursor: versionName ? 'pointer' : 'not-allowed',
            }}
            disabled={!versionName}
            onClick={onClose}
          >
            Create & Lock Parameters
          </button>
          <button
            className="px-4 py-2 rounded-lg text-sm"
            style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
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
   Pack Card (List View)
   ======================================================================== */

function PackCard({ pack, onToggle, onDetails, onCopy, hasVariations, isExpanded, onToggleExpand }: {
  pack: TfPack;
  onToggle: () => void;
  onDetails: () => void;
  onCopy: () => void;
  hasVariations?: boolean;
  isExpanded?: boolean;
  onToggleExpand?: () => void;
}) {
  const isVariation = !pack.isDefault;

  return (
    <div
      className="flex items-center gap-3 px-4 py-3 rounded-xl border transition-colors"
      style={{
        background: 'var(--bg-card)',
        borderColor: 'var(--border)',
        boxShadow: 'var(--card-shadow)',
        backdropFilter: 'var(--card-backdrop)',
        WebkitBackdropFilter: 'var(--card-backdrop)',
        opacity: pack.enabled ? 1 : 0.6,
        marginLeft: isVariation ? 24 : 0,
      }}
    >
      {/* Expand chevron for packs with variations */}
      {hasVariations ? (
        <button
          onClick={(e) => { e.stopPropagation(); onToggleExpand?.(); }}
          className="w-5 h-5 rounded flex items-center justify-center text-xs flex-shrink-0 transition-transform"
          style={{ color: 'var(--text-muted)', transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)' }}
        >
          {'\u25B6'}
        </button>
      ) : isVariation ? (
        <div className="w-5 flex-shrink-0" />
      ) : null}

      <Toggle enabled={pack.enabled} onChange={onToggle} />

      <div className="flex-1 min-w-0">
        {/* Row 1: Name, version, tags, counts */}
        <div className="flex items-center gap-2 mb-1 flex-wrap">
          <span className="font-semibold text-sm" style={{ color: 'var(--text-primary)' }}>{pack.name}</span>
          <span
            className="text-[10px] px-1.5 py-0.5 rounded"
            style={{
              color: pack.isDefault ? 'var(--text-muted)' : 'var(--text-secondary)',
              background: 'var(--bg-input)',
            }}
          >
            {pack.version}
          </span>
          {pack.tags.map((tag) => <TagBadge key={tag} tag={tag} />)}
          <span
            className="text-[10px] cursor-default"
            style={{ color: 'var(--text-muted)' }}
            title={pack.outputs.length > 0 ? pack.outputs.map((o) => `${o.code} — ${o.description}`).join('\n') : 'No states'}
          >
            {pack.outputs.length} states
          </span>
          <span
            className="text-[10px] cursor-default"
            style={{ color: 'var(--text-muted)' }}
            title={pack.triggers.map((t) => t.name).join('\n')}
          >
            {pack.triggers.length} trigger{pack.triggers.length !== 1 ? 's' : ''}
          </span>
        </div>
        {/* Row 2: All locked parameters — indicator params + exec type summary */}
        <div className="flex items-center gap-x-3 gap-y-0.5 flex-wrap text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>
          {pack.params.map((p) => {
            const short = p.label.replace('Period', '').replace('Threshold', '').replace('Multiplier', 'x').replace('Inner Band ', '').replace('Outer Band ', '').replace('Bar Count', 'Bars').trim();
            return (
              <span key={p.key}>
                {short}:<span style={{ color: 'var(--text-secondary)' }}>{p.value}</span>
              </span>
            );
          })}
          <span style={{ color: 'var(--border)' }}>|</span>
          {(['C', 'L', 'LC', 'CC'] as const).map((et) => {
            const config = pack.triggers[0]?.execVariants[et];
            if (!config?.enabled) return (
              <span key={et} style={{ opacity: 0.4 }}>[{et}] off</span>
            );
            const ref = config.referenceBar === -1 ? '-1' : '0';
            const ord = config.orderType === 'market' ? 'mkt' : 'lmt';
            return (
              <span key={et}>
                [{et}] ref:<span style={{ color: 'var(--text-secondary)' }}>{ref}</span>{' '}
                {ord}
              </span>
            );
          })}
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-2 flex-shrink-0">
        <button onClick={onDetails} className="px-3 py-1.5 rounded text-xs font-medium" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
          Details
        </button>
        <button onClick={onCopy} className="px-3 py-1.5 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>
          Create Variation
        </button>
      </div>
    </div>
  );
}

/* ========================================================================
   Main Component
   ======================================================================== */

/* ========================================================================
   Detail View — Trigger Parameters Tab
   ======================================================================== */

const EXEC_TYPE_LABELS: Record<string, { label: string; description: string }> = {
  C:  { label: 'Close', description: 'Enter when bar closes. One-stage, most reliable.' },
  L:  { label: 'Level', description: 'Enter on intra-bar level cross. Fastest fill. Requires HiFi for backtest.' },
  LC: { label: 'Level-Close', description: 'Level cross entry, bar close confirmation. Two-stage with bail on non-confirm.' },
  CC: { label: 'Close-Close', description: 'Bar close entry, next bar close confirmation. Most conservative two-stage.' },
};

function StartingTriggerParamsTab({ pack }: { pack: TfPack }) {
  const configFor = (et: 'C' | 'L' | 'LC' | 'CC') =>
    pack.triggers[0]?.execVariants[et] || { enabled: true, referenceBar: 0, orderType: 'market' as const };
  const isDisabled = pack.isSaved;
  const selectStyle = (disabled: boolean) => ({
    background: 'var(--bg-input)', border: '1px solid var(--border)',
    color: disabled ? 'var(--text-muted)' : 'var(--text-primary)',
  });

  const cConfig = configFor('C');
  const lConfig = configFor('L');
  const lcConfig = configFor('LC');
  const ccConfig = configFor('CC');

  return (
    <div className="space-y-4">
      <Card>
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Default Execution Type Settings</h3>
          {pack.isSaved && (
            <span className="text-xs px-2 py-1 rounded-lg flex items-center gap-1.5" style={{ background: 'var(--orange-muted)', color: 'var(--orange)' }}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <rect x="3" y="11" width="18" height="11" rx="2" /><path d="M7 11V7a5 5 0 0110 0v4" />
              </svg>
              Locked after save
            </span>
          )}
        </div>
        <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
          Configure how each execution type behaves for all triggers in this pack.
          Two-stage types (LC, CC) inherit their first-stage settings from the corresponding one-stage type.
        </p>

        {!pack.isSaved && (
          <div className="mb-4 px-3 py-2 rounded-lg text-xs flex items-start gap-2" style={{ background: 'var(--orange-muted)', color: 'var(--orange)' }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="flex-shrink-0 mt-0.5">
              <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
              <line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" />
            </svg>
            These settings will be locked permanently when the variation is saved.
          </div>
        )}

        <div className="space-y-5">
          {/* ---- [C] Close ---- */}
          <div className="rounded-xl border p-4" style={{ borderColor: 'var(--border)', background: 'var(--bg-secondary)' }}>
            <div className="flex items-center gap-3 mb-3">
              <ExecBadge exec="C" />
              <div className="flex-1">
                <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>Close</span>
                <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{EXEC_TYPE_LABELS.C.description}</p>
              </div>
              <label className="flex items-center gap-2 cursor-pointer">
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Enabled</span>
                <Toggle enabled={cConfig.enabled} onChange={() => {}} />
              </label>
            </div>
            {cConfig.enabled && (
              <div className="grid grid-cols-2 lg:grid-cols-3 gap-3 mt-3">
                <div>
                  <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>Reference Bar</label>
                  <select className="w-full px-2 py-1.5 rounded text-xs" style={selectStyle(isDisabled)} value={cConfig.referenceBar} disabled={isDisabled}>
                    <option value={0}>Current bar (0)</option>
                    <option value={-1}>Previous bar (-1)</option>
                  </select>
                </div>
                <div>
                  <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>Order Type</label>
                  <select className="w-full px-2 py-1.5 rounded text-xs" style={selectStyle(isDisabled)} value={cConfig.orderType} disabled={isDisabled}>
                    <option value="market">Market</option>
                    <option value="limit">Limit</option>
                  </select>
                </div>
              </div>
            )}
          </div>

          {/* ---- [L] Level ---- */}
          <div className="rounded-xl border p-4" style={{ borderColor: 'var(--border)', background: 'var(--bg-secondary)' }}>
            <div className="flex items-center gap-3 mb-3">
              <ExecBadge exec="L" />
              <div className="flex-1">
                <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>Level</span>
                <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{EXEC_TYPE_LABELS.L.description}</p>
              </div>
              <label className="flex items-center gap-2 cursor-pointer">
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Enabled</span>
                <Toggle enabled={lConfig.enabled} onChange={() => {}} />
              </label>
            </div>
            {lConfig.enabled && (
              <div className="grid grid-cols-2 lg:grid-cols-3 gap-3 mt-3">
                <div>
                  <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>Reference Bar</label>
                  <select className="w-full px-2 py-1.5 rounded text-xs" style={selectStyle(isDisabled)} value={lConfig.referenceBar} disabled={isDisabled}>
                    <option value={-1}>Previous bar (-1)</option>
                    <option value={0}>Current bar (0)</option>
                  </select>
                </div>
                <div>
                  <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>Order Type</label>
                  <select className="w-full px-2 py-1.5 rounded text-xs" style={selectStyle(isDisabled)} value={lConfig.orderType} disabled={isDisabled}>
                    <option value="market">Market</option>
                    <option value="limit">Limit</option>
                  </select>
                </div>
                <div>
                  <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>Hold Seconds</label>
                  <input type="number" className="w-full px-2 py-1.5 rounded text-xs font-mono" style={selectStyle(isDisabled)} value={lConfig.holdSeconds ?? 0} min={0} disabled={isDisabled} />
                </div>
              </div>
            )}
          </div>

          {/* ---- [LC] Level-Close ---- */}
          <div className="rounded-xl border p-4" style={{ borderColor: 'var(--border)', background: 'var(--bg-secondary)' }}>
            <div className="flex items-center gap-3 mb-3">
              <ExecBadge exec="LC" />
              <div className="flex-1">
                <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>Level-Close</span>
                <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{EXEC_TYPE_LABELS.LC.description}</p>
              </div>
              <label className="flex items-center gap-2 cursor-pointer">
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Enabled</span>
                <Toggle enabled={lcConfig.enabled} onChange={() => {}} />
              </label>
            </div>
            {lcConfig.enabled && (
              <div className="mt-3 space-y-3">
                {/* Inherited from [L] */}
                <div className="px-3 py-2 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-[10px] font-medium" style={{ color: 'var(--text-secondary)' }}>Stage 1: Level Cross</span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ color: 'var(--blue)', background: 'var(--blue-muted)' }}>
                      inherits from [L]
                    </span>
                  </div>
                  <div className="flex gap-4 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                    <span>ref: <span className="font-mono">{lConfig.referenceBar === -1 ? 'prev bar' : 'current bar'}</span></span>
                    <span>order: <span className="font-mono">{lConfig.orderType}</span></span>
                    <span>hold: <span className="font-mono">{lConfig.holdSeconds ?? 0}s</span></span>
                  </div>
                </div>
                {/* Stage 2: Confirmation (unique to LC) */}
                <div>
                  <span className="text-[10px] font-medium block mb-2" style={{ color: 'var(--text-secondary)' }}>Stage 2: Bar Close Confirmation</span>
                  <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
                    <div>
                      <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>Confirm Bar Offset</label>
                      <select className="w-full px-2 py-1.5 rounded text-xs" style={selectStyle(isDisabled)} value={lcConfig.confirmBarOffset ?? 0} disabled={isDisabled}>
                        <option value={0}>Same bar (0)</option>
                        <option value={1}>Next bar (+1)</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>Bail Action (on non-confirm)</label>
                      <select className="w-full px-2 py-1.5 rounded text-xs" style={selectStyle(isDisabled)} value={lcConfig.bailAction ?? 'exit_market'} disabled={isDisabled}>
                        <option value="exit_market">Immediate market exit</option>
                        <option value="exit_limit">Immediate limit exit</option>
                        <option value="exit_limit_breakeven">Limit exit at breakeven</option>
                      </select>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* ---- [CC] Close-Close ---- */}
          <div className="rounded-xl border p-4" style={{ borderColor: 'var(--border)', background: 'var(--bg-secondary)' }}>
            <div className="flex items-center gap-3 mb-3">
              <ExecBadge exec="CC" />
              <div className="flex-1">
                <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>Close-Close</span>
                <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{EXEC_TYPE_LABELS.CC.description}</p>
              </div>
              <label className="flex items-center gap-2 cursor-pointer">
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Enabled</span>
                <Toggle enabled={ccConfig.enabled} onChange={() => {}} />
              </label>
            </div>
            {ccConfig.enabled && (
              <div className="mt-3 space-y-3">
                {/* Inherited from [C] */}
                <div className="px-3 py-2 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-[10px] font-medium" style={{ color: 'var(--text-secondary)' }}>Stage 1: Bar Close Entry</span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ color: 'var(--green)', background: 'var(--green-muted)' }}>
                      inherits from [C]
                    </span>
                  </div>
                  <div className="flex gap-4 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                    <span>ref: <span className="font-mono">{cConfig.referenceBar === -1 ? 'prev bar' : 'current bar'}</span></span>
                    <span>order: <span className="font-mono">{cConfig.orderType}</span></span>
                  </div>
                </div>
                {/* Stage 2: Confirmation (unique to CC) */}
                <div>
                  <span className="text-[10px] font-medium block mb-2" style={{ color: 'var(--text-secondary)' }}>Stage 2: Next Bar Confirmation</span>
                  <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
                    <div>
                      <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>Confirm Bar Offset</label>
                      <select className="w-full px-2 py-1.5 rounded text-xs" style={selectStyle(isDisabled)} value={ccConfig.confirmBarOffset ?? 1} disabled={isDisabled}>
                        <option value={1}>Next bar (+1)</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>Bail Action (on non-confirm)</label>
                      <select className="w-full px-2 py-1.5 rounded text-xs" style={selectStyle(isDisabled)} value={ccConfig.bailAction ?? 'exit_market'} disabled={isDisabled}>
                        <option value="exit_market">Immediate market exit</option>
                        <option value="exit_limit">Immediate limit exit</option>
                        <option value="exit_limit_breakeven">Limit exit at breakeven</option>
                      </select>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </Card>
    </div>
  );
}

/* ========================================================================
   Main Component
   ======================================================================== */

export default function TfConfluencePage() {
  // ---- API hooks (MUST come before any early returns) ----
  const { data: apiGroups, isLoading: groupsLoading, error: groupsError } = useConfluenceGroups();
  const { data: apiTemplates, isLoading: templatesLoading } = useConfluenceTemplates();
  const saveMutation = useSaveConfluenceGroups();

  // Map API data → V5 TfPack[]
  const apiPacks = useMemo<TfPack[]>(() => {
    if (!apiGroups) return [];
    return apiGroups.map((g) => mapApiGroupToTfPack(g, apiTemplates ?? undefined));
  }, [apiGroups, apiTemplates]);

  const [packs, setPacks] = useState<TfPack[]>([]);
  const [detailPack, setDetailPack] = useState<TfPack | null>(null);
  const [draftPack, setDraftPack] = useState<TfPack | null>(null); // unsaved variation being configured
  const [search, setSearch] = useState('');
  const [expandedTemplates, setExpandedTemplates] = useState<Set<string>>(new Set());

  // Sync API data into local state when it arrives (once)
  const hasInitialized = useRef(false);
  useEffect(() => {
    if (apiPacks.length > 0 && !hasInitialized.current) {
      hasInitialized.current = true;
      setPacks(apiPacks);
      const keys = apiPacks.filter((p) => !p.isDefault).map((p) => p.templateKey);
      setExpandedTemplates(new Set(keys));
    }
  }, [apiPacks]);

  const isLoading = groupsLoading || templatesLoading;
  const error = groupsError;

  const enabledCount = packs.filter((p) => p.enabled).length;

  // ---- Loading / Error states ----

  if (isLoading) {
    return (
      <div>
        <PageHeader title="TF Confluence Packs" subtitle="Loading..." />
        <div className="space-y-3 mt-4">
          {[1, 2, 3, 4, 5].map((i) => (
            <Card key={i}>
              <div className="animate-pulse space-y-3">
                <div className="h-4 rounded w-1/3" style={{ background: 'var(--border)' }} />
                <div className="h-3 rounded w-2/3" style={{ background: 'var(--border)' }} />
                <div className="h-8 rounded" style={{ background: 'var(--bg-input)' }} />
              </div>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div>
        <PageHeader title="TF Confluence Packs" subtitle="Error loading packs" />
        <Card>
          <div className="text-center py-8" style={{ color: 'var(--red)' }}>
            Failed to load confluence packs. Check your connection and try again.
          </div>
        </Card>
      </div>
    );
  }

  function togglePack(id: string) {
    setPacks((prev) => prev.map((p) => (p.id === id ? { ...p, enabled: !p.enabled } : p)));
  }

  function toggleTemplate(key: string) {
    setExpandedTemplates((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  // Group packs: default first, then variations underneath
  const groupedPacks = useMemo(() => {
    const groups: { templateKey: string; default: TfPack; variations: TfPack[] }[] = [];
    const templateOrder: string[] = [];

    for (const pack of packs) {
      if (pack.isDefault) {
        templateOrder.push(pack.templateKey);
        groups.push({ templateKey: pack.templateKey, default: pack, variations: [] });
      }
    }

    for (const pack of packs) {
      if (!pack.isDefault) {
        const group = groups.find((g) => g.templateKey === pack.templateKey);
        if (group) group.variations.push(pack);
      }
    }

    return groups;
  }, [packs]);

  // Filter by search
  const filteredGroups = useMemo(() => {
    if (!search.trim()) return groupedPacks;
    const q = search.toLowerCase();
    return groupedPacks
      .map((group) => ({
        ...group,
        default: group.default,
        variations: group.variations.filter(
          (v) => v.name.toLowerCase().includes(q) || v.version.toLowerCase().includes(q) || v.tags.some((t) => t.toLowerCase().includes(q))
        ),
      }))
      .filter(
        (group) =>
          group.default.name.toLowerCase().includes(q) ||
          group.default.tags.some((t) => t.toLowerCase().includes(q)) ||
          group.variations.length > 0
      );
  }, [groupedPacks, search]);

  // The active pack for the detail view — either a saved pack or an unsaved draft
  const activePack = draftPack || detailPack;
  const isDraft = draftPack !== null;

  function handleCopy(source: TfPack) {
    // Create a draft variation from the source — opens detail view in editable mode
    const draft: TfPack = {
      ...source,
      id: `${source.templateKey}-draft-${Date.now()}`,
      version: '',
      isDefault: false,
      isSaved: false,
    };
    setDraftPack(draft);
    setDetailPack(null);
  }

  function handleSaveVariation() {
    if (!draftPack || !draftPack.version.trim()) return;
    const saved: TfPack = { ...draftPack, isSaved: true, id: `${draftPack.templateKey}-${draftPack.version.toLowerCase().replace(/\s+/g, '-')}` };
    setPacks((prev) => [...prev, saved]);
    setDraftPack(null);
    setDetailPack(saved);
  }

  function handleDiscardDraft() {
    setDraftPack(null);
  }

  /* ---- Detail View (saved or draft) ---- */
  if (activePack) {
    return (
      <div>
        <PageHeader
          title={isDraft ? `New Variation of ${activePack.name}` : `${activePack.name} (${activePack.version})`}
          subtitle={isDraft ? 'Configure parameters, trigger settings, and preview before saving' : activePack.tags.join(' \u00b7 ')}
          backHref="#"
          actions={
            <div className="flex items-center gap-3">
              {isDraft && (
                <>
                  {/* Variation name input */}
                  <div className="flex items-center gap-2">
                    <label className="text-xs" style={{ color: 'var(--text-muted)' }}>Name:</label>
                    <input
                      className="px-2 py-1.5 rounded-lg text-sm w-40"
                      style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                      placeholder="e.g. Scalping"
                      value={activePack.version}
                      onChange={(e) => setDraftPack({ ...activePack, version: e.target.value })}
                    />
                  </div>
                  <button
                    onClick={handleSaveVariation}
                    className="px-4 py-2 rounded-lg text-sm font-medium"
                    style={{
                      background: 'var(--accent)',
                      color: 'white',
                      opacity: activePack.version.trim() ? 1 : 0.5,
                      cursor: activePack.version.trim() ? 'pointer' : 'not-allowed',
                    }}
                    disabled={!activePack.version.trim()}
                  >
                    Save as Variation
                  </button>
                  <button
                    onClick={handleDiscardDraft}
                    className="px-3 py-1.5 rounded-lg text-xs"
                    style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
                  >
                    Discard
                  </button>
                </>
              )}
              {!isDraft && (
                <>
                  <button
                    onClick={() => handleCopy(activePack)}
                    className="px-3 py-1.5 rounded-lg text-xs"
                    style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
                  >
                    Copy as Variation
                  </button>
                  <button
                    onClick={() => setDetailPack(null)}
                    className="px-4 py-2 rounded-lg text-sm"
                    style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
                  >
                    Back to Packs
                  </button>
                </>
              )}
            </div>
          }
        />

        {isDraft && (
          <div className="mb-4 px-4 py-3 rounded-xl flex items-start gap-2" style={{ background: 'var(--orange-muted)', border: '1px solid var(--orange)' }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="flex-shrink-0 mt-0.5" style={{ color: 'var(--orange)' }}>
              <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
              <line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" />
            </svg>
            <div>
              <p className="text-xs font-medium" style={{ color: 'var(--orange)' }}>Draft mode</p>
              <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                Indicator parameters and execution type settings will be <strong>permanently locked</strong> once saved.
                Use the Preview tab to test your configuration before saving. Plot settings can be changed at any time.
              </p>
            </div>
          </div>
        )}

        <TabBar tabs={['Parameters', 'Trigger Parameters', 'Plot Settings', 'Outputs & Triggers', 'Preview', 'Code', 'Danger Zone']}>
          {(tab) => (
            <div>
              {tab === 'Parameters' && <ParametersTab pack={activePack} />}
              {tab === 'Trigger Parameters' && <StartingTriggerParamsTab pack={activePack} />}
              {tab === 'Plot Settings' && <PlotSettingsTab pack={activePack} />}
              {tab === 'Outputs & Triggers' && <OutputsTriggersTab pack={activePack} />}
              {tab === 'Preview' && <PreviewTab pack={activePack} />}
              {tab === 'Code' && <CodeTab pack={activePack} />}
              {tab === 'Danger Zone' && <DangerZoneTab pack={activePack} />}
            </div>
          )}
        </TabBar>
      </div>
    );
  }

  /* ---- List View ---- */
  return (
    <div>
      <PageHeader
        title="TF Confluence Packs"
        subtitle="Timeframe-specific indicators that compute on each bar"
        actions={
          <Link
            href="/confluence-packs/pack-builder"
            className="px-4 py-2 rounded-lg text-sm font-medium inline-flex items-center gap-2"
            style={{ background: 'var(--accent)', color: 'white' }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="12" y1="5" x2="12" y2="19" />
              <line x1="5" y1="12" x2="19" y2="12" />
            </svg>
            Create New Template
          </Link>
        }
      />

      {/* Stats + Search */}
      <div className="flex items-center gap-4 mb-5">
        <p className="text-sm flex-shrink-0" style={{ color: 'var(--text-muted)' }}>
          {packs.length} packs, {enabledCount} enabled
        </p>
        <div className="flex-1 relative">
          <svg
            width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
            className="absolute left-3 top-1/2 -translate-y-1/2"
            style={{ color: 'var(--text-muted)' }}
          >
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <input
            className="w-full pl-9 pr-3 py-2 rounded-lg text-sm"
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
            placeholder="Search packs by name, version, or tag..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
      </div>

      {/* Pack list — grouped by template with expandable variations */}
      <div className="space-y-2">
        {filteredGroups.map((group) => {
          const hasVariations = group.variations.length > 0;
          const isExpanded = expandedTemplates.has(group.templateKey);

          return (
            <div key={group.templateKey}>
              {/* Default pack card */}
              <PackCard
                pack={group.default}
                onToggle={() => togglePack(group.default.id)}
                onDetails={() => setDetailPack(group.default)}
                onCopy={() => handleCopy(group.default)}
                hasVariations={hasVariations}
                isExpanded={isExpanded}
                onToggleExpand={() => toggleTemplate(group.templateKey)}
              />

              {/* Variation count hint when collapsed */}
              {hasVariations && !isExpanded && (
                <button
                  onClick={() => toggleTemplate(group.templateKey)}
                  className="ml-10 mt-1 text-[10px]"
                  style={{ color: 'var(--text-muted)' }}
                >
                  {group.variations.length} variation{group.variations.length !== 1 ? 's' : ''}
                </button>
              )}

              {/* Variations (nested, shown when expanded) */}
              {isExpanded && group.variations.map((variation) => (
                <div key={variation.id} className="mt-2">
                  <PackCard
                    pack={variation}
                    onToggle={() => togglePack(variation.id)}
                    onDetails={() => setDetailPack(variation)}
                    onCopy={() => handleCopy(variation)}
                  />
                </div>
              ))}
            </div>
          );
        })}

        {filteredGroups.length === 0 && (
          <div className="text-center py-12">
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No packs match &quot;{search}&quot;</p>
          </div>
        )}
      </div>

    </div>
  );
}
