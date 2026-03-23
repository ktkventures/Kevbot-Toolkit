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
  type: 'int' | 'float' | 'bool' | 'select';
  default: number | boolean | string;
  min?: number;
  max?: number;
  options?: string[];
  help?: string;
}

interface OutputDef {
  code: string;
  description: string;
}

interface TriggerDef {
  id: string;
  base: string;
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
  outputs: OutputDef[];
  triggers: TriggerDef[];
  triggerPrefix: string;
  conditionLogic: string;
}

interface PackInstance {
  id: string;
  templateKey: string;
  versionName: string;
  enabled: boolean;
  isDefault: boolean;
  paramValues: Record<string, number | boolean | string>;
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
  execType: string;
}

/* ========================================================================
   Template Definitions (real data from general_packs.py)
   ======================================================================== */

const templates: Record<string, TemplateDef> = {
  time_of_day: {
    name: 'Time of Day',
    category: 'Time',
    description: 'Filter trades to a specific time-of-day window',
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
      { id: 'tod_window_open', base: 'window_open', name: 'Window Opens', direction: 'BOTH', type: 'ENTRY', execution: '[C]' },
      { id: 'tod_window_close', base: 'window_close', name: 'Window Closes', direction: 'BOTH', type: 'EXIT', execution: '[C]' },
    ],
    triggerPrefix: 'tod',
    conditionLogic: 'time_window',
  },

  trading_session: {
    name: 'Trading Session',
    category: 'Time',
    description: 'Filter trades by market session (pre-market, regular, after-hours)',
    parameters: [
      { key: 'session', label: 'Session', type: 'select', default: 'regular', options: ['pre_market', 'regular', 'after_hours', 'extended'] },
    ],
    outputs: [
      { code: 'IN_SESSION', description: 'Current time is within the selected session' },
      { code: 'OUT_OF_SESSION', description: 'Current time is outside the selected session' },
    ],
    triggers: [
      { id: 'session_session_open', base: 'session_open', name: 'Session Opens', direction: 'BOTH', type: 'ENTRY', execution: '[C]' },
      { id: 'session_session_close', base: 'session_close', name: 'Session Closes', direction: 'BOTH', type: 'EXIT', execution: '[C]' },
    ],
    triggerPrefix: 'session',
    conditionLogic: 'session_filter',
  },

  day_of_week: {
    name: 'Day of Week',
    category: 'Calendar',
    description: 'Allow or block trading on specific days of the week',
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
    triggerPrefix: 'dow',
    conditionLogic: 'day_filter',
  },

  calendar_filter: {
    name: 'Calendar Filter',
    category: 'Calendar',
    description: 'Block trading around high-impact economic events',
    parameters: [
      { key: 'avoid_fomc', label: 'Avoid FOMC Days', type: 'bool', default: true },
      { key: 'avoid_opex', label: 'Avoid OpEx Days', type: 'bool', default: false },
      { key: 'avoid_nfp', label: 'Avoid NFP Days', type: 'bool', default: true },
      { key: 'buffer_minutes', label: 'Buffer (minutes)', type: 'int', default: 30, min: 0, max: 120, help: 'Buffer around event time (proxy mode uses full-day blocks)' },
    ],
    outputs: [
      { code: 'CLEAR', description: 'No high-impact events -- trading allowed' },
      { code: 'BLOCKED', description: 'High-impact event window -- trading blocked' },
    ],
    triggers: [
      { id: 'cal_event_block_start', base: 'event_block_start', name: 'Event Block Starts', direction: 'BOTH', type: 'EXIT', execution: '[C]' },
      { id: 'cal_event_clear', base: 'event_clear', name: 'Event Block Clears', direction: 'BOTH', type: 'ENTRY', execution: '[C]' },
    ],
    triggerPrefix: 'cal',
    conditionLogic: 'calendar_filter',
  },
};

/* ========================================================================
   Mock Pack Instances
   ======================================================================== */

const initialPacks: PackInstance[] = [
  {
    id: 'tod_default',
    templateKey: 'time_of_day',
    versionName: 'Default',
    enabled: true,
    isDefault: true,
    paramValues: { start_hour: 9, start_minute: 30, end_hour: 12, end_minute: 0 },
  },
  {
    id: 'tod_power_hour',
    templateKey: 'time_of_day',
    versionName: 'Power Hour',
    enabled: true,
    isDefault: false,
    paramValues: { start_hour: 15, start_minute: 0, end_hour: 16, end_minute: 0 },
  },
  {
    id: 'session_default',
    templateKey: 'trading_session',
    versionName: 'Default',
    enabled: true,
    isDefault: true,
    paramValues: { session: 'regular' },
  },
  {
    id: 'dow_default',
    templateKey: 'day_of_week',
    versionName: 'Default',
    enabled: true,
    isDefault: true,
    paramValues: { monday: true, tuesday: true, wednesday: true, thursday: true, friday: true },
  },
  {
    id: 'cal_default',
    templateKey: 'calendar_filter',
    versionName: 'Default',
    enabled: false,
    isDefault: true,
    paramValues: { avoid_fomc: true, avoid_opex: false, avoid_nfp: true, buffer_minutes: 30 },
  },
];

/* ========================================================================
   Mock Preview Data
   ======================================================================== */

const mockStateTimelines: Record<string, StateTimelineEntry[]> = {
  time_of_day: [
    { time: '09:00:00', state: 'OUT_OF_WINDOW', prevState: 'OUT_OF_WINDOW' },
    { time: '09:30:00', state: 'IN_WINDOW', prevState: 'OUT_OF_WINDOW' },
    { time: '10:00:00', state: 'IN_WINDOW', prevState: 'IN_WINDOW' },
    { time: '11:00:00', state: 'IN_WINDOW', prevState: 'IN_WINDOW' },
    { time: '12:00:00', state: 'OUT_OF_WINDOW', prevState: 'IN_WINDOW' },
    { time: '13:00:00', state: 'OUT_OF_WINDOW', prevState: 'OUT_OF_WINDOW' },
    { time: '14:00:00', state: 'OUT_OF_WINDOW', prevState: 'OUT_OF_WINDOW' },
  ],
  trading_session: [
    { time: '04:00:00', state: 'OUT_OF_SESSION', prevState: 'OUT_OF_SESSION' },
    { time: '04:01:00', state: 'IN_SESSION', prevState: 'OUT_OF_SESSION' },
    { time: '09:29:00', state: 'IN_SESSION', prevState: 'IN_SESSION' },
    { time: '09:30:00', state: 'OUT_OF_SESSION', prevState: 'IN_SESSION' },
    { time: '16:00:00', state: 'OUT_OF_SESSION', prevState: 'OUT_OF_SESSION' },
  ],
  day_of_week: [
    { time: 'Mon 09:30', state: 'ALLOWED_DAY', prevState: 'ALLOWED_DAY' },
    { time: 'Tue 09:30', state: 'ALLOWED_DAY', prevState: 'ALLOWED_DAY' },
    { time: 'Wed 09:30', state: 'ALLOWED_DAY', prevState: 'ALLOWED_DAY' },
    { time: 'Thu 09:30', state: 'ALLOWED_DAY', prevState: 'ALLOWED_DAY' },
    { time: 'Fri 09:30', state: 'ALLOWED_DAY', prevState: 'ALLOWED_DAY' },
  ],
  calendar_filter: [
    { time: 'Mon 09:30', state: 'CLEAR', prevState: 'CLEAR' },
    { time: 'Tue 09:30', state: 'CLEAR', prevState: 'CLEAR' },
    { time: 'Wed 14:00', state: 'BLOCKED', prevState: 'CLEAR' },
    { time: 'Wed 14:30', state: 'BLOCKED', prevState: 'BLOCKED' },
    { time: 'Wed 15:00', state: 'CLEAR', prevState: 'BLOCKED' },
    { time: 'Thu 09:30', state: 'CLEAR', prevState: 'CLEAR' },
    { time: 'Fri 08:30', state: 'BLOCKED', prevState: 'CLEAR' },
    { time: 'Fri 09:00', state: 'CLEAR', prevState: 'BLOCKED' },
  ],
};

const mockTriggerEvents: Record<string, TriggerEvent[]> = {
  time_of_day: [
    { time: '09:30:00', trigger: 'Window Opens', direction: 'BOTH', type: 'ENTRY', execType: '[C]' },
    { time: '12:00:00', trigger: 'Window Closes', direction: 'BOTH', type: 'EXIT', execType: '[C]' },
  ],
  trading_session: [
    { time: '04:00:00', trigger: 'Session Opens', direction: 'BOTH', type: 'ENTRY', execType: '[C]' },
    { time: '09:30:00', trigger: 'Session Closes', direction: 'BOTH', type: 'EXIT', execType: '[C]' },
  ],
  day_of_week: [],
  calendar_filter: [
    { time: 'Wed 14:00', trigger: 'Event Block Starts', direction: 'BOTH', type: 'EXIT', execType: '[C]' },
    { time: 'Wed 15:00', trigger: 'Event Block Clears', direction: 'BOTH', type: 'ENTRY', execType: '[C]' },
    { time: 'Fri 08:30', trigger: 'Event Block Starts', direction: 'BOTH', type: 'EXIT', execType: '[C]' },
    { time: 'Fri 09:00', trigger: 'Event Block Clears', direction: 'BOTH', type: 'ENTRY', execType: '[C]' },
  ],
};

/* ========================================================================
   Style Constants
   ======================================================================== */

const categoryColors: Record<string, { color: string; bg: string }> = {
  'Time': { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  'Calendar': { color: 'var(--orange)', bg: 'var(--orange-muted)' },
};

const execTypeBadgeStyles: Record<string, { color: string; bg: string }> = {
  '[C]': { color: 'var(--green)', bg: 'var(--green-muted)' },
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

const stateColors: Record<string, { color: string; bg: string }> = {
  'IN_WINDOW': { color: 'var(--green)', bg: 'var(--green-muted)' },
  'OUT_OF_WINDOW': { color: 'var(--red)', bg: 'var(--red-muted)' },
  'IN_SESSION': { color: 'var(--green)', bg: 'var(--green-muted)' },
  'OUT_OF_SESSION': { color: 'var(--red)', bg: 'var(--red-muted)' },
  'ALLOWED_DAY': { color: 'var(--green)', bg: 'var(--green-muted)' },
  'BLOCKED_DAY': { color: 'var(--red)', bg: 'var(--red-muted)' },
  'CLEAR': { color: 'var(--green)', bg: 'var(--green-muted)' },
  'BLOCKED': { color: 'var(--red)', bg: 'var(--red-muted)' },
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
  return tpl.parameters.map((p) => {
    const val = pack.paramValues[p.key];
    if (p.type === 'bool') return `${p.label}: ${val ? 'Yes' : 'No'}`;
    return `${p.label}: ${val}`;
  }).join(', ');
}

function formatTimeValue(hour: number | boolean | string, minute: number | boolean | string): string {
  const h = Number(hour);
  const m = Number(minute);
  const period = h >= 12 ? 'PM' : 'AM';
  const displayHour = h > 12 ? h - 12 : h === 0 ? 12 : h;
  return `${displayHour}:${String(m).padStart(2, '0')} ${period}`;
}

function getCategoriesWithPacks(packs: PackInstance[]): { category: string; packs: PackInstance[] }[] {
  const categoryOrder = ['Time', 'Calendar'];
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

function StateBadge({ state }: { state: string }) {
  const s = stateColors[state] || { color: 'var(--text-muted)', bg: 'var(--bg-input)' };
  return (
    <span className="text-xs font-mono px-2 py-0.5 rounded font-medium" style={{ color: s.color, background: s.bg }}>
      {state}
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
          Template: {tpl.name} | Logic: {tpl.conditionLogic}
        </span>
      </div>

      <div className="space-y-4">
        {tpl.parameters.map((param) => (
          <div key={param.key}>
            <div className="flex items-center gap-4">
              <label className="text-sm w-52 flex-shrink-0" style={{ color: 'var(--text-secondary)' }}>
                {param.label}
              </label>

              {param.type === 'bool' ? (
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    defaultChecked={pack.paramValues[param.key] as boolean}
                    className="w-4 h-4 rounded"
                  />
                  <span className="text-sm" style={{ color: 'var(--text-primary)' }}>
                    {pack.paramValues[param.key] ? 'Enabled' : 'Disabled'}
                  </span>
                </label>
              ) : param.type === 'select' ? (
                <select
                  className="px-3 py-2 rounded-lg text-sm flex-1"
                  style={selectStyle}
                  defaultValue={pack.paramValues[param.key] as string}
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
                {param.type === 'bool' && `Toggle | Default: ${param.default ? 'On' : 'Off'}`}
                {param.type === 'select' && `Select | Options: ${param.options?.join(', ')} | Default: ${param.default}`}
                {param.help ? ` -- ${param.help}` : ''}
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Time preview for time_of_day template */}
      {pack.templateKey === 'time_of_day' && (
        <div
          className="mt-5 px-4 py-3 rounded-lg"
          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
        >
          <span className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>
            Window Preview:
          </span>
          <span className="text-sm font-mono ml-2" style={{ color: 'var(--text-primary)' }}>
            {formatTimeValue(pack.paramValues.start_hour, pack.paramValues.start_minute)} -{' '}
            {formatTimeValue(pack.paramValues.end_hour, pack.paramValues.end_minute)} ET
          </span>
        </div>
      )}

      {/* Session label for trading_session template */}
      {pack.templateKey === 'trading_session' && (
        <div
          className="mt-5 px-4 py-3 rounded-lg"
          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
        >
          <span className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>
            Session Hours:
          </span>
          <span className="text-sm font-mono ml-2" style={{ color: 'var(--text-primary)' }}>
            {pack.paramValues.session === 'regular' && '9:30 AM - 4:00 PM ET'}
            {pack.paramValues.session === 'pre_market' && '4:00 AM - 9:30 AM ET'}
            {pack.paramValues.session === 'after_hours' && '4:00 PM - 8:00 PM ET'}
            {pack.paramValues.session === 'extended' && '4:00 AM - 8:00 PM ET'}
          </span>
        </div>
      )}

      {/* Day summary for day_of_week template */}
      {pack.templateKey === 'day_of_week' && (
        <div
          className="mt-5 px-4 py-3 rounded-lg"
          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
        >
          <span className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>
            Active Days:
          </span>
          <div className="flex items-center gap-2 mt-2">
            {['monday', 'tuesday', 'wednesday', 'thursday', 'friday'].map((day) => (
              <span
                key={day}
                className="text-xs px-2 py-1 rounded font-mono"
                style={{
                  background: pack.paramValues[day] ? 'var(--green-muted)' : 'var(--red-muted)',
                  color: pack.paramValues[day] ? 'var(--green)' : 'var(--red)',
                }}
              >
                {day.slice(0, 3).toUpperCase()}
              </span>
            ))}
          </div>
        </div>
      )}

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
   Detail: Outputs & Triggers Tab
   ======================================================================== */

function OutputsTriggersTab({ tpl }: { tpl: TemplateDef }) {
  return (
    <div className="grid grid-cols-2 gap-6">
      {/* Outputs column */}
      <Card>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
            Condition Outputs
          </h3>
          <span
            className="text-xs px-2 py-0.5 rounded"
            style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}
          >
            {tpl.outputs.length} states
          </span>
        </div>
        <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
          Conditions are evaluated on every bar and produce a categorical state. These states
          gate strategy entries via confluence requirements.
        </p>
        <div className="space-y-2">
          {tpl.outputs.map((output) => (
            <div
              key={output.code}
              className="px-3 py-2.5 rounded-lg"
              style={{ background: 'var(--bg-input)' }}
            >
              <div className="flex items-center gap-2 mb-1">
                <StateBadge state={output.code} />
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
            {tpl.triggers.length} trigger{tpl.triggers.length !== 1 ? 's' : ''}
          </span>
        </div>
        {tpl.triggers.length > 0 ? (
          <>
            <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
              Triggers fire on state transitions and can be used as entry/exit signals.
              Prefix: <code className="font-mono px-1 py-0.5 rounded" style={{ background: 'var(--bg-input)' }}>{tpl.triggerPrefix}</code>
            </p>
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
          </>
        ) : (
          <div
            className="px-4 py-8 rounded-lg text-center"
            style={{ background: 'var(--bg-input)' }}
          >
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              No triggers available
            </p>
            <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
              This pack provides conditions only. States are used as confluence
              gates -- no entry/exit signals are generated.
            </p>
          </div>
        )}
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
          label={`${symbol} price chart with ${tpl.name} condition overlay${showConditions ? ' + state bands' : ''}${showTriggers ? ' + trigger markers' : ''}`}
          height={380}
        />
      </Card>

      {/* Data tables below chart */}
      <div className="grid grid-cols-2 gap-4">
        {/* State Timeline */}
        <Card>
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
              Condition States
            </h4>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
              {stateTimeline.length} entries
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
                  <StateBadge state={row.state} />
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
              {triggerEvents.length} event{triggerEvents.length !== 1 ? 's' : ''}
            </span>
          </div>
          {triggerEvents.length > 0 ? (
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
                  gridTemplateColumns: '80px 1fr 55px 50px 35px',
                  gap: '4px',
                }}
              >
                <span>Time</span>
                <span>Trigger</span>
                <span>Dir</span>
                <span>Type</span>
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
                    gridTemplateColumns: '80px 1fr 55px 50px 35px',
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
                  <ExecBadge exec={row.execType} />
                </div>
              ))}
            </div>
          ) : (
            <div
              className="px-4 py-8 rounded-lg text-center"
              style={{ background: 'var(--bg-input)' }}
            >
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                No triggers -- this pack provides conditions only.
              </p>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}

/* ========================================================================
   Detail: Code Tab
   ======================================================================== */

function CodeTab({ pack, tpl }: { pack: PackInstance; tpl: TemplateDef }) {
  const [showConditionCode, setShowConditionCode] = useState(false);
  const [showTriggerCode, setShowTriggerCode] = useState(false);

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

  const paramEntries = tpl.parameters.map((p) => {
    const val = pack.paramValues[p.key];
    if (p.type === 'bool') return `${p.key}=${val ? 'True' : 'False'}`;
    if (p.type === 'select') return `${p.key}="${val}"`;
    return `${p.key}=${val}`;
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
            {tpl.parameters.map((p) => (
              <div key={p.key} className="flex justify-between">
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{p.label}</span>
                <span className="text-xs font-mono font-semibold" style={{ color: 'var(--text-primary)' }}>
                  {p.type === 'bool' ? (pack.paramValues[p.key] ? 'True' : 'False') : String(pack.paramValues[p.key])}
                </span>
              </div>
            ))}
          </div>
        </div>
        <div className="mt-3 flex gap-2">
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
            Condition Column:
          </span>
          <code
            className="text-xs font-mono px-2 py-0.5 rounded"
            style={{ background: 'var(--bg-input)', color: 'var(--accent)' }}
          >
            GP_{pack.id.toUpperCase()}
          </code>
        </div>
      </Card>

      {/* Condition Evaluation Function */}
      <Card>
        <button
          className="flex items-center justify-between w-full text-left"
          onClick={() => setShowConditionCode(!showConditionCode)}
        >
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
            Condition Evaluation Function
          </h3>
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {showConditionCode ? 'Collapse' : 'Expand'}
          </span>
        </button>
        {showConditionCode && (
          <div className="mt-3">
            <pre style={codeBlockStyle}>
{`def evaluate_condition(df, pack):
    """
    Evaluate ${tpl.name} condition on each bar.
    Logic: ${tpl.conditionLogic}

    Parameters:
${tpl.parameters.map((p) => `        ${p.key}: ${p.label} (${p.type})`).join('\n')}

    Outputs:
${tpl.outputs.map((o) => `        ${o.code}: ${o.description}`).join('\n')}

    Returns pd.Series of condition labels.
    """
    # Implementation in general_packs.py -> _eval_${tpl.conditionLogic}()
    ...`}
            </pre>
          </div>
        )}
      </Card>

      {/* Trigger Detection Function */}
      {tpl.triggers.length > 0 && (
        <Card>
          <button
            className="flex items-center justify-between w-full text-left"
            onClick={() => setShowTriggerCode(!showTriggerCode)}
          >
            <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
              Trigger Detection Function
            </h3>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
              {showTriggerCode ? 'Collapse' : 'Expand'}
            </span>
          </button>
          {showTriggerCode && (
            <div className="mt-3">
              <pre style={codeBlockStyle}>
{`def detect_triggers(df, pack):
    """
    Detect ${tpl.name} trigger events.
    Prefix: ${tpl.triggerPrefix}

    Triggers:
${tpl.triggers.map((t) => `        ${t.id}: ${t.name} (${t.direction} ${t.type})`).join('\n')}

    Returns dict of {trigger_column: pd.Series(bool)}.
    """
    # Implementation in general_packs.py -> detect_triggers()
    ...`}
              </pre>
            </div>
          )}
        </Card>
      )}

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

      <TabBar tabs={['Parameters', 'Outputs & Triggers', 'Preview', 'Code', 'Danger Zone']}>
        {(tab) => (
          <div>
            {tab === 'Parameters' && <ParametersTab pack={pack} tpl={tpl} />}
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
            {tpl.outputs.map((output) => (
              <StateBadge key={output.code} state={output.code} />
            ))}
          </div>
        </div>

        {/* Right side: trigger count + actions */}
        <div className="flex flex-col items-end gap-2 flex-shrink-0">
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {tpl.triggers.length > 0
              ? `${tpl.triggers.length} trigger${tpl.triggers.length !== 1 ? 's' : ''}`
              : 'conditions only'}
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
    <Modal title="Create New General Pack" isOpen={isOpen} onClose={onClose} width="560px">
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
            {tpl.outputs.map((o) => (
              <StateBadge key={o.code} state={o.code} />
            ))}
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
            placeholder="e.g. NY Open, Power Hour, Conservative..."
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
            {tpl.parameters.map((p) => {
              const def = p.type === 'bool' ? (p.default ? 'On' : 'Off') : String(p.default);
              return `${p.label}: ${def}`;
            }).join('  |  ')}
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

export default function GeneralPacksV2() {
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
    const paramValues: Record<string, number | boolean | string> = {};
    for (const p of tpl.parameters) paramValues[p.key] = p.default;

    const newPack: PackInstance = {
      id: newId,
      templateKey,
      versionName,
      enabled: true,
      isDefault: false,
      paramValues,
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
        title="General Packs"
        subtitle="Strategy-wide conditions and optional triggers (time of day, sessions, calendar)"
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
