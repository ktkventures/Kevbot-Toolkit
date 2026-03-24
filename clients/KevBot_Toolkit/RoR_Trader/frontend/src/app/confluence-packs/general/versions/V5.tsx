'use client';

import { useState, useMemo } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import ChartPlaceholder from '@/components/ChartPlaceholder';
import Link from 'next/link';

/* ========================================================================
   Types
   ======================================================================== */

interface PackParam {
  key: string;
  label: string;
  type: 'int' | 'float' | 'bool' | 'select';
  value: number | boolean | string;
  default: number | boolean | string;
  min?: number;
  max?: number;
  options?: { value: string; label: string }[];
}

interface OutputDef {
  code: string;
  description: string;
}

interface PackTrigger {
  id: string;
  name: string;
  sentiment: 'bullish' | 'bearish' | 'neutral';
  description: string;
}

interface GeneralPack {
  id: string;
  templateKey: string;
  name: string;
  version: string;
  tags: string[];
  enabled: boolean;
  isDefault: boolean;
  isSaved: boolean;
  conditionLogic: string;
  params: PackParam[];
  outputs: OutputDef[];
  triggers: PackTrigger[];
}

/* ========================================================================
   Template Definitions (mirrors general_packs.py)
   ======================================================================== */

interface TemplateDef {
  name: string;
  tags: string[];
  description: string;
  conditionLogic: string;
  parameters: Omit<PackParam, 'value'>[];
  outputs: OutputDef[];
  triggers: PackTrigger[];
}

const TEMPLATES: Record<string, TemplateDef> = {
  time_of_day: {
    name: 'Time of Day',
    tags: ['Time'],
    description: 'Filter trades to a specific time-of-day window',
    conditionLogic: 'time_window',
    parameters: [
      { key: 'start_hour', label: 'Start Hour', type: 'int', default: 9, min: 0, max: 23 },
      { key: 'start_minute', label: 'Start Minute', type: 'int', default: 30, min: 0, max: 59 },
      { key: 'end_hour', label: 'End Hour', type: 'int', default: 12, min: 0, max: 23 },
      { key: 'end_minute', label: 'End Minute', type: 'int', default: 0, min: 0, max: 59 },
    ],
    outputs: [
      { code: 'IN_WINDOW', description: 'Current time is within the configured window' },
      { code: 'OUT_OF_WINDOW', description: 'Current time is outside the configured window' },
    ],
    triggers: [
      { id: 'window_open', name: 'Window Opens', sentiment: 'bullish', description: 'Fires when time enters the configured window' },
      { id: 'window_close', name: 'Window Closes', sentiment: 'bearish', description: 'Fires when time exits the configured window' },
    ],
  },
  trading_session: {
    name: 'Trading Session',
    tags: ['Time', 'Session'],
    description: 'Filter trades to a specific market session',
    conditionLogic: 'session_filter',
    parameters: [
      { key: 'session', label: 'Session', type: 'select', default: 'regular', options: [
        { value: 'pre_market', label: 'Pre-Market (4:00\u201309:30 ET)' },
        { value: 'regular', label: 'Regular Hours (9:30\u201316:00 ET)' },
        { value: 'after_hours', label: 'After Hours (16:00\u201320:00 ET)' },
        { value: 'extended', label: 'Extended (4:00\u201320:00 ET)' },
      ]},
    ],
    outputs: [
      { code: 'IN_SESSION', description: 'Current time is within the selected session' },
      { code: 'OUT_OF_SESSION', description: 'Current time is outside the selected session' },
    ],
    triggers: [
      { id: 'session_open', name: 'Session Opens', sentiment: 'bullish', description: 'Fires when market enters the selected session' },
      { id: 'session_close', name: 'Session Closes', sentiment: 'bearish', description: 'Fires when market exits the selected session' },
    ],
  },
  day_of_week: {
    name: 'Day of Week',
    tags: ['Calendar'],
    description: 'Allow or block trading on specific weekdays',
    conditionLogic: 'day_filter',
    parameters: [
      { key: 'monday', label: 'Monday', type: 'bool', default: true },
      { key: 'tuesday', label: 'Tuesday', type: 'bool', default: true },
      { key: 'wednesday', label: 'Wednesday', type: 'bool', default: true },
      { key: 'thursday', label: 'Thursday', type: 'bool', default: true },
      { key: 'friday', label: 'Friday', type: 'bool', default: true },
    ],
    outputs: [
      { code: 'ALLOWED_DAY', description: 'Today is an allowed trading day' },
      { code: 'BLOCKED_DAY', description: 'Today is a blocked trading day' },
    ],
    triggers: [],
  },
  calendar_filter: {
    name: 'Calendar Filter',
    tags: ['Calendar', 'Events'],
    description: 'Avoid trading around major economic events (FOMC, NFP, OpEx)',
    conditionLogic: 'calendar_filter',
    parameters: [
      { key: 'avoid_fomc', label: 'Avoid FOMC Days', type: 'bool', default: true },
      { key: 'avoid_nfp', label: 'Avoid NFP Days', type: 'bool', default: true },
      { key: 'avoid_opex', label: 'Avoid OpEx Days', type: 'bool', default: false },
      { key: 'buffer_minutes', label: 'Buffer (minutes)', type: 'int', default: 30, min: 0, max: 120 },
    ],
    outputs: [
      { code: 'CLEAR', description: 'No conflicting events \u2014 trading allowed' },
      { code: 'BLOCKED', description: 'Event detected \u2014 trading blocked' },
    ],
    triggers: [
      { id: 'event_block_start', name: 'Event Block Starts', sentiment: 'bearish', description: 'Fires when an event block period begins' },
      { id: 'event_clear', name: 'Event Clears', sentiment: 'bullish', description: 'Fires when event block period ends' },
    ],
  },
};

/* ========================================================================
   Mock Pack Instances
   ======================================================================== */

function createDefaultPack(templateKey: string, id: string, version: string): GeneralPack {
  const t = TEMPLATES[templateKey];
  return {
    id,
    templateKey,
    name: t.name,
    version,
    tags: t.tags,
    enabled: true,
    isDefault: true,
    isSaved: true,
    conditionLogic: t.conditionLogic,
    params: t.parameters.map((p) => ({ ...p, value: p.default })),
    outputs: t.outputs,
    triggers: t.triggers,
  };
}

const initialPacks: GeneralPack[] = [
  createDefaultPack('time_of_day', 'tod-ny-open', 'NY Open'),
  createDefaultPack('trading_session', 'session-regular', 'Regular Hours'),
  createDefaultPack('day_of_week', 'dow-weekdays', 'All Weekdays'),
  { ...createDefaultPack('calendar_filter', 'cal-avoid-fomc-nfp', 'Avoid FOMC & NFP'), enabled: false },
  {
    ...createDefaultPack('time_of_day', 'tod-power-hour', 'Power Hour'),
    isDefault: false,
    params: [
      { key: 'start_hour', label: 'Start Hour', type: 'int', value: 15, default: 9, min: 0, max: 23 },
      { key: 'start_minute', label: 'Start Minute', type: 'int', value: 0, default: 30, min: 0, max: 59 },
      { key: 'end_hour', label: 'End Hour', type: 'int', value: 16, default: 12, min: 0, max: 23 },
      { key: 'end_minute', label: 'End Minute', type: 'int', value: 0, default: 0, min: 0, max: 59 },
    ],
  },
  {
    ...createDefaultPack('day_of_week', 'dow-midweek', 'Midweek Only'),
    isDefault: false,
    enabled: false,
    params: [
      { key: 'monday', label: 'Monday', type: 'bool', value: false, default: true },
      { key: 'tuesday', label: 'Tuesday', type: 'bool', value: true, default: true },
      { key: 'wednesday', label: 'Wednesday', type: 'bool', value: true, default: true },
      { key: 'thursday', label: 'Thursday', type: 'bool', value: true, default: true },
      { key: 'friday', label: 'Friday', type: 'bool', value: false, default: true },
    ],
  },
];

/* ========================================================================
   Style Constants
   ======================================================================== */

const tagColors: Record<string, { color: string; bg: string }> = {
  Time: { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  Session: { color: 'var(--accent)', bg: 'var(--accent-muted)' },
  Calendar: { color: 'var(--orange)', bg: 'var(--orange-muted)' },
  Events: { color: 'var(--red)', bg: 'var(--red-muted)' },
};

/* ========================================================================
   Badge & Toggle Components
   ======================================================================== */

function TagBadge({ tag }: { tag: string }) {
  const style = tagColors[tag] || { color: 'var(--text-muted)', bg: 'var(--bg-input)' };
  return <span className="text-[10px] font-medium px-1.5 py-0.5 rounded" style={{ color: style.color, background: style.bg }}>{tag}</span>;
}

function SentimentBadge({ sentiment }: { sentiment: string }) {
  const styles: Record<string, { color: string; bg: string; icon: string }> = {
    bullish: { color: 'var(--green)', bg: 'var(--green-muted)', icon: '\u2191' },
    bearish: { color: 'var(--red)', bg: 'var(--red-muted)', icon: '\u2193' },
    neutral: { color: 'var(--text-muted)', bg: 'var(--bg-input)', icon: '\u2194' },
  };
  const s = styles[sentiment] || styles.neutral;
  return <span className="text-[10px] font-medium px-1.5 py-0.5 rounded" style={{ color: s.color, background: s.bg }}>{s.icon} {sentiment}</span>;
}

function Toggle({ enabled, onChange }: { enabled: boolean; onChange: () => void }) {
  return (
    <button
      onClick={(e) => { e.stopPropagation(); onChange(); }}
      className="w-9 h-5 rounded-full relative flex-shrink-0 transition-colors"
      style={{ background: enabled ? 'var(--accent)' : 'var(--bg-input)', border: enabled ? 'none' : '1px solid var(--border)' }}
    >
      <div className="w-3.5 h-3.5 rounded-full absolute transition-all" style={{ background: enabled ? 'white' : 'var(--text-muted)', top: '3px', left: enabled ? '19px' : '3px' }} />
    </button>
  );
}

/* ========================================================================
   Detail View — Parameters Tab
   ======================================================================== */

function ParametersTab({ pack }: { pack: GeneralPack }) {
  const template = TEMPLATES[pack.templateKey];
  return (
    <Card>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Pack Parameters</h3>
        {pack.isSaved && (
          <span className="text-xs px-2 py-1 rounded-lg flex items-center gap-1.5" style={{ background: 'var(--orange-muted)', color: 'var(--orange)' }}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <rect x="3" y="11" width="18" height="11" rx="2" /><path d="M7 11V7a5 5 0 0110 0v4" />
            </svg>
            Parameters locked after save
          </span>
        )}
      </div>

      {pack.isSaved && (
        <div className="mb-4 px-3 py-2 rounded-lg text-xs" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>
          Parameters are locked to protect active strategies and portfolios. To use different parameters, create a new variation via the Create Variation button.
        </div>
      )}

      <div className="space-y-3">
        {pack.params.map((param) => (
          <div key={param.key} className="flex items-center gap-4">
            <label className="text-sm w-40 flex-shrink-0" style={{ color: 'var(--text-secondary)' }}>{param.label}</label>

            {param.type === 'bool' ? (
              <Toggle enabled={param.value as boolean} onChange={() => {}} />
            ) : param.type === 'select' ? (
              <select
                className="px-3 py-2 rounded-lg text-sm flex-1"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: pack.isSaved ? 'var(--text-muted)' : 'var(--text-primary)' }}
                value={param.value as string}
                disabled={pack.isSaved}
              >
                {param.options?.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
              </select>
            ) : (
              <input
                type="number"
                className="w-24 px-3 py-2 rounded-lg text-sm text-center font-mono"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: pack.isSaved ? 'var(--text-muted)' : 'var(--text-primary)', cursor: pack.isSaved ? 'not-allowed' : 'text' }}
                value={param.value as number}
                disabled={pack.isSaved}
                readOnly={pack.isSaved}
              />
            )}

            {param.type === 'int' && param.min !== undefined && (
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>({param.min}\u2013{param.max})</span>
            )}
            {param.value !== param.default && param.type !== 'bool' && (
              <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ color: 'var(--orange)', background: 'var(--orange-muted)' }}>
                default: {String(param.default)}
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
              <line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" />
            </svg>
            Once saved, parameters cannot be changed. Review all tabs and use the &quot;Save as Variation&quot; button in the header when ready.
          </div>
        </div>
      )}

      {template && (
        <div className="mt-6 pt-4 border-t" style={{ borderColor: 'var(--border)' }}>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            Template: <span className="font-medium" style={{ color: 'var(--text-secondary)' }}>{template.name}</span>
            {' \u00b7 '}{template.description}
            {' \u00b7 '}Evaluator: <span className="font-mono">{template.conditionLogic}</span>
          </p>
        </div>
      )}
    </Card>
  );
}

/* ========================================================================
   Detail View — Outputs & Triggers Tab
   ======================================================================== */

function OutputsTriggersTab({ pack }: { pack: GeneralPack }) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Conditions / Outputs */}
      <Card>
        <div className="flex items-center gap-3 mb-4">
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Condition States</h3>
          <span className="flex-1" />
          <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>
            {pack.outputs.length} states
          </span>
        </div>
        <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
          Scalar condition evaluated on each bar&apos;s timestamp. Used as a strategy-wide gate for entries and exits.
        </p>
        <div className="space-y-2">
          {pack.outputs.map((output) => (
            <div key={output.code} className="flex items-start gap-3 px-3 py-2.5 rounded-lg" style={{ background: 'var(--bg-input)' }}>
              <span className="text-sm font-mono font-bold flex-shrink-0 mt-0.5" style={{ color: 'var(--text-primary)' }}>{output.code}</span>
              <span className="text-xs leading-relaxed" style={{ color: 'var(--text-muted)' }}>{output.description}</span>
            </div>
          ))}
        </div>
      </Card>

      {/* Triggers */}
      <Card>
        <div className="flex items-center gap-3 mb-4">
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Triggers</h3>
          <span className="flex-1" />
          <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>
            {pack.triggers.length} trigger{pack.triggers.length !== 1 ? 's' : ''}
          </span>
        </div>
        {pack.triggers.length === 0 ? (
          <p className="text-xs px-3 py-6 text-center" style={{ color: 'var(--text-muted)' }}>
            This pack has no triggers. It acts as a condition-only gate &mdash; strategies reference its output states directly as confluence conditions.
          </p>
        ) : (
          <>
            <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
              Triggers fire on state transitions. All general pack triggers use bar-close evaluation.
            </p>
            <div className="space-y-2">
              {pack.triggers.map((trigger) => (
                <div key={trigger.id} className="px-3 py-2.5 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm font-medium flex-1" style={{ color: 'var(--text-primary)' }}>{trigger.name}</span>
                    <SentimentBadge sentiment={trigger.sentiment} />
                  </div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{trigger.description}</p>
                  <p className="text-[10px] font-mono mt-1" style={{ color: 'var(--text-muted)', opacity: 0.6 }}>{pack.id}_{trigger.id}</p>
                </div>
              ))}
            </div>
          </>
        )}
      </Card>
    </div>
  );
}

/* ========================================================================
   Detail View — Preview Tab
   ======================================================================== */

interface StateTimelineEntry { time: string; state: string; prevState: string }

const mockStateTimelines: Record<string, StateTimelineEntry[]> = {
  time_of_day: [
    { time: '09:00:00', state: 'OUT_OF_WINDOW', prevState: 'OUT_OF_WINDOW' },
    { time: '09:30:00', state: 'IN_WINDOW', prevState: 'OUT_OF_WINDOW' },
    { time: '10:00:00', state: 'IN_WINDOW', prevState: 'IN_WINDOW' },
    { time: '11:00:00', state: 'IN_WINDOW', prevState: 'IN_WINDOW' },
    { time: '12:00:00', state: 'OUT_OF_WINDOW', prevState: 'IN_WINDOW' },
    { time: '13:00:00', state: 'OUT_OF_WINDOW', prevState: 'OUT_OF_WINDOW' },
  ],
  trading_session: [
    { time: '04:00:00', state: 'OUT_OF_SESSION', prevState: 'OUT_OF_SESSION' },
    { time: '09:30:00', state: 'IN_SESSION', prevState: 'OUT_OF_SESSION' },
    { time: '12:00:00', state: 'IN_SESSION', prevState: 'IN_SESSION' },
    { time: '16:00:00', state: 'OUT_OF_SESSION', prevState: 'IN_SESSION' },
  ],
  day_of_week: [
    { time: 'Monday', state: 'ALLOWED_DAY', prevState: 'BLOCKED_DAY' },
    { time: 'Tuesday', state: 'ALLOWED_DAY', prevState: 'ALLOWED_DAY' },
    { time: 'Wednesday', state: 'ALLOWED_DAY', prevState: 'ALLOWED_DAY' },
    { time: 'Thursday', state: 'ALLOWED_DAY', prevState: 'ALLOWED_DAY' },
    { time: 'Friday', state: 'ALLOWED_DAY', prevState: 'ALLOWED_DAY' },
  ],
  calendar_filter: [
    { time: '2024-03-06', state: 'CLEAR', prevState: 'CLEAR' },
    { time: '2024-03-07', state: 'BLOCKED', prevState: 'CLEAR' },
    { time: '2024-03-08', state: 'CLEAR', prevState: 'BLOCKED' },
    { time: '2024-03-14', state: 'CLEAR', prevState: 'CLEAR' },
    { time: '2024-03-15', state: 'BLOCKED', prevState: 'CLEAR' },
    { time: '2024-03-18', state: 'CLEAR', prevState: 'BLOCKED' },
  ],
};

function PreviewTab({ pack }: { pack: GeneralPack }) {
  const stateTimeline = mockStateTimelines[pack.templateKey] || [];

  return (
    <div className="space-y-4">
      <Card>
        <ChartPlaceholder
          label={`${pack.name} (${pack.version}) condition bands over time`}
          height={280}
        />
      </Card>

      <Card>
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Condition State Timeline</h4>
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{stateTimeline.length} entries</span>
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
        </div>
      </Card>
    </div>
  );
}

/* ========================================================================
   Detail View — Code Tab
   ======================================================================== */

function CodeTab({ pack }: { pack: GeneralPack }) {
  const [showEvaluator, setShowEvaluator] = useState(false);

  const codeBlockStyle: React.CSSProperties = {
    background: 'var(--bg-primary)', border: '1px solid var(--border)', color: 'var(--text-secondary)',
    fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.6', padding: '16px', borderRadius: '8px',
    whiteSpace: 'pre-wrap', overflowX: 'auto',
  };

  return (
    <div className="space-y-4">
      <Card>
        <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Active Parameters</h3>
        <div className="rounded-lg px-4 py-3" style={{ background: 'var(--bg-primary)', border: '1px solid var(--border)' }}>
          <div className="grid grid-cols-2 gap-y-2 gap-x-8">
            {pack.params.map((p) => (
              <div key={p.key} className="flex justify-between">
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{p.label}</span>
                <span className="text-xs font-mono font-semibold" style={{ color: 'var(--text-primary)' }}>{String(p.value)}</span>
              </div>
            ))}
          </div>
        </div>
      </Card>

      <Card>
        <button className="flex items-center justify-between w-full text-left" onClick={() => setShowEvaluator(!showEvaluator)}>
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Evaluator Function</h3>
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{showEvaluator ? 'Collapse' : 'Expand'}</span>
        </button>
        {showEvaluator && (
          <div className="mt-3">
            <pre style={codeBlockStyle}>
{`def evaluate_${pack.conditionLogic}(timestamp, ${pack.params.map((p) => `${p.key}=${JSON.stringify(p.value)}`).join(', ')}):
    """
    Scalar condition evaluator for ${pack.name}.
    Logic: ${pack.conditionLogic}

    Parameters:
${pack.params.map((p) => `        ${p.key}: ${p.label} (${p.type})`).join('\n')}

    Returns: "${pack.outputs.map((o) => o.code).join('" or "')}"
    """
    # Implementation in general_packs.py
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

function DangerZoneTab({ pack }: { pack: GeneralPack }) {
  return (
    <Card>
      <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--red)' }}>Danger Zone</h3>
      <div className="space-y-6">
        <div>
          <label className="text-sm mb-2 block" style={{ color: 'var(--text-secondary)' }}>Rename Variation</label>
          <div className="flex gap-3">
            <input
              className="px-3 py-2 rounded-lg text-sm flex-1"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: pack.isDefault ? 'var(--text-muted)' : 'var(--text-primary)', cursor: pack.isDefault ? 'not-allowed' : 'text' }}
              defaultValue={pack.version}
              disabled={pack.isDefault}
            />
            <button className="px-4 py-2 rounded-lg text-sm" style={{ background: pack.isDefault ? 'var(--bg-input)' : 'var(--bg-card)', border: '1px solid var(--border)', color: pack.isDefault ? 'var(--text-muted)' : 'var(--text-secondary)', cursor: pack.isDefault ? 'not-allowed' : 'pointer' }} disabled={pack.isDefault}>Rename</button>
          </div>
          {pack.isDefault && <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>Default packs cannot be renamed.</p>}
        </div>
        <div className="pt-4 border-t" style={{ borderColor: 'var(--border)' }}>
          <p className="text-sm mb-2" style={{ color: 'var(--text-secondary)' }}>Delete this variation permanently.</p>
          <button className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: pack.isDefault ? 'var(--bg-input)' : 'var(--red)', color: pack.isDefault ? 'var(--text-muted)' : 'white', cursor: pack.isDefault ? 'not-allowed' : 'pointer' }} disabled={pack.isDefault}>Delete Variation</button>
          {pack.isDefault && <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>Default packs cannot be deleted.</p>}
        </div>
      </div>
    </Card>
  );
}

/* ========================================================================
   Pack Card (List View)
   ======================================================================== */

function formatParamSummary(pack: GeneralPack): string {
  return pack.params.map((p) => {
    if (p.type === 'bool') return `${p.label.replace('Avoid ', '').replace(' Days', '')}: ${p.value ? 'Yes' : 'No'}`;
    if (p.type === 'select') {
      const opt = p.options?.find((o) => o.value === p.value);
      return opt ? opt.label.split('(')[0].trim() : String(p.value);
    }
    const short = p.label.replace('Start ', 'S').replace('End ', 'E').replace('Hour', 'h').replace('Minute', 'm').replace('Buffer (minutes)', 'Buffer');
    return `${short}:${p.value}`;
  }).join(', ');
}

function PackCard({ pack, onToggle, onDetails, onCopy, hasVariations, isExpanded, onToggleExpand }: {
  pack: GeneralPack;
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
      style={{ background: 'var(--bg-card)', borderColor: 'var(--border)', boxShadow: 'var(--card-shadow)', backdropFilter: 'var(--card-backdrop)', WebkitBackdropFilter: 'var(--card-backdrop)', opacity: pack.enabled ? 1 : 0.6, marginLeft: isVariation ? 24 : 0 }}
    >
      {hasVariations ? (
        <button onClick={(e) => { e.stopPropagation(); onToggleExpand?.(); }} className="w-5 h-5 rounded flex items-center justify-center text-xs flex-shrink-0 transition-transform" style={{ color: 'var(--text-muted)', transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)' }}>
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
          <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ color: pack.isDefault ? 'var(--text-muted)' : 'var(--text-secondary)', background: 'var(--bg-input)' }}>{pack.version}</span>
          {pack.tags.map((tag) => <TagBadge key={tag} tag={tag} />)}
          <span className="text-[10px] cursor-default" style={{ color: 'var(--text-muted)' }} title={pack.outputs.map((o) => `${o.code} \u2014 ${o.description}`).join('\n')}>
            {pack.outputs.length} states
          </span>
          <span className="text-[10px] cursor-default" style={{ color: 'var(--text-muted)' }} title={pack.triggers.length > 0 ? pack.triggers.map((t) => t.name).join('\n') : 'No triggers (condition-only)'}>
            {pack.triggers.length} trigger{pack.triggers.length !== 1 ? 's' : ''}
          </span>
        </div>
        {/* Row 2: Parameter summary */}
        <p className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>
          {formatParamSummary(pack)}
        </p>
      </div>

      {/* Actions */}
      <div className="flex gap-2 flex-shrink-0">
        <button onClick={onDetails} className="px-3 py-1.5 rounded text-xs font-medium" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>Details</button>
        <button onClick={onCopy} className="px-3 py-1.5 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>Create Variation</button>
      </div>
    </div>
  );
}

/* ========================================================================
   Main Component
   ======================================================================== */

export default function GeneralPacksV5() {
  const [packs, setPacks] = useState<GeneralPack[]>(initialPacks);
  const [detailPack, setDetailPack] = useState<GeneralPack | null>(null);
  const [draftPack, setDraftPack] = useState<GeneralPack | null>(null);
  const [search, setSearch] = useState('');
  const [expandedTemplates, setExpandedTemplates] = useState<Set<string>>(() => {
    const keys = initialPacks.filter((p) => !p.isDefault).map((p) => p.templateKey);
    return new Set(Array.from(new Set(keys)));
  });

  const enabledCount = packs.filter((p) => p.enabled).length;

  function togglePack(id: string) {
    setPacks((prev) => prev.map((p) => (p.id === id ? { ...p, enabled: !p.enabled } : p)));
  }

  function toggleTemplate(key: string) {
    setExpandedTemplates((prev) => { const next = new Set(prev); if (next.has(key)) next.delete(key); else next.add(key); return next; });
  }

  const groupedPacks = useMemo(() => {
    const groups: { templateKey: string; default: GeneralPack; variations: GeneralPack[] }[] = [];
    for (const pack of packs) {
      if (pack.isDefault) groups.push({ templateKey: pack.templateKey, default: pack, variations: [] });
    }
    for (const pack of packs) {
      if (!pack.isDefault) {
        const group = groups.find((g) => g.templateKey === pack.templateKey);
        if (group) group.variations.push(pack);
      }
    }
    return groups;
  }, [packs]);

  const filteredGroups = useMemo(() => {
    if (!search.trim()) return groupedPacks;
    const q = search.toLowerCase();
    return groupedPacks
      .map((group) => ({ ...group, variations: group.variations.filter((v) => v.name.toLowerCase().includes(q) || v.version.toLowerCase().includes(q) || v.tags.some((t) => t.toLowerCase().includes(q))) }))
      .filter((group) => group.default.name.toLowerCase().includes(q) || group.default.tags.some((t) => t.toLowerCase().includes(q)) || group.variations.length > 0);
  }, [groupedPacks, search]);

  const activePack = draftPack || detailPack;
  const isDraft = draftPack !== null;

  function handleCopy(source: GeneralPack) {
    const draft: GeneralPack = { ...source, id: `${source.templateKey}-draft-${Date.now()}`, version: '', isDefault: false, isSaved: false };
    setDraftPack(draft);
    setDetailPack(null);
  }

  function handleSaveVariation() {
    if (!draftPack || !draftPack.version.trim()) return;
    const saved: GeneralPack = { ...draftPack, isSaved: true, id: `${draftPack.templateKey}-${draftPack.version.toLowerCase().replace(/\s+/g, '-')}` };
    setPacks((prev) => [...prev, saved]);
    setDraftPack(null);
    setDetailPack(saved);
  }

  /* ---- Detail View ---- */
  if (activePack) {
    return (
      <div>
        <PageHeader
          title={isDraft ? `New Variation of ${activePack.name}` : `${activePack.name} (${activePack.version})`}
          subtitle={isDraft ? 'Configure parameters and preview before saving' : activePack.tags.join(' \u00b7 ')}
          backHref="#"
          actions={
            <div className="flex items-center gap-3">
              {isDraft ? (
                <>
                  <div className="flex items-center gap-2">
                    <label className="text-xs" style={{ color: 'var(--text-muted)' }}>Name:</label>
                    <input className="px-2 py-1.5 rounded-lg text-sm w-40" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} placeholder="e.g. Power Hour" value={activePack.version} onChange={(e) => setDraftPack({ ...activePack, version: e.target.value })} />
                  </div>
                  <button onClick={handleSaveVariation} className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--accent)', color: 'white', opacity: activePack.version.trim() ? 1 : 0.5, cursor: activePack.version.trim() ? 'pointer' : 'not-allowed' }} disabled={!activePack.version.trim()}>
                    Save as Variation
                  </button>
                  <button onClick={() => setDraftPack(null)} className="px-3 py-1.5 rounded-lg text-xs" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>Discard</button>
                </>
              ) : (
                <>
                  <button onClick={() => handleCopy(activePack)} className="px-3 py-1.5 rounded-lg text-xs" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>Create Variation</button>
                  <button onClick={() => setDetailPack(null)} className="px-4 py-2 rounded-lg text-sm" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>Back to Packs</button>
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
                Parameters will be <strong>permanently locked</strong> once saved.
                Use the Preview tab to verify your configuration before saving.
              </p>
            </div>
          </div>
        )}

        <TabBar tabs={['Parameters', 'Outputs & Triggers', 'Preview', 'Code', 'Danger Zone']}>
          {(tab) => (
            <div>
              {tab === 'Parameters' && <ParametersTab pack={activePack} />}
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
        title="General Confluence Packs"
        subtitle="Strategy-wide conditions that gate when trades are allowed"
        actions={
          <Link href="/confluence-packs/pack-builder" className="px-4 py-2 rounded-lg text-sm font-medium inline-flex items-center gap-2" style={{ background: 'var(--accent)', color: 'white' }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" /></svg>
            Create New Template
          </Link>
        }
      />

      <div className="flex items-center gap-4 mb-5">
        <p className="text-sm flex-shrink-0" style={{ color: 'var(--text-muted)' }}>{packs.length} packs, {enabledCount} enabled</p>
        <div className="flex-1 relative">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: 'var(--text-muted)' }}>
            <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <input className="w-full pl-9 pr-3 py-2 rounded-lg text-sm" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} placeholder="Search packs by name, version, or tag..." value={search} onChange={(e) => setSearch(e.target.value)} />
        </div>
      </div>

      <div className="space-y-2">
        {filteredGroups.map((group) => {
          const hasVariations = group.variations.length > 0;
          const isExpanded = expandedTemplates.has(group.templateKey);
          return (
            <div key={group.templateKey}>
              <PackCard pack={group.default} onToggle={() => togglePack(group.default.id)} onDetails={() => setDetailPack(group.default)} onCopy={() => handleCopy(group.default)} hasVariations={hasVariations} isExpanded={isExpanded} onToggleExpand={() => toggleTemplate(group.templateKey)} />
              {hasVariations && !isExpanded && (
                <button onClick={() => toggleTemplate(group.templateKey)} className="ml-10 mt-1 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                  {group.variations.length} variation{group.variations.length !== 1 ? 's' : ''}
                </button>
              )}
              {isExpanded && group.variations.map((variation) => (
                <div key={variation.id} className="mt-2">
                  <PackCard pack={variation} onToggle={() => togglePack(variation.id)} onDetails={() => setDetailPack(variation)} onCopy={() => handleCopy(variation)} />
                </div>
              ))}
            </div>
          );
        })}
        {filteredGroups.length === 0 && (
          <div className="text-center py-12"><p className="text-sm" style={{ color: 'var(--text-muted)' }}>No packs match &quot;{search}&quot;</p></div>
        )}
      </div>
    </div>
  );
}
