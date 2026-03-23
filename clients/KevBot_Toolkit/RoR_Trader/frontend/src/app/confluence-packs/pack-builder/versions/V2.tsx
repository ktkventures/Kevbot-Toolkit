'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import ChartPlaceholder from '@/components/ChartPlaceholder';

/* ========================================================================
   Types
   ======================================================================== */

interface ParamDef {
  id: string;
  name: string;
  type: 'int' | 'float';
  default: number;
  min: number;
  max: number;
}

interface OutputState {
  id: string;
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

interface ValidationItem {
  label: string;
  status: 'pass' | 'fail' | 'warn';
  message: string;
}

/* ========================================================================
   Mock Data & Constants
   ======================================================================== */

const initialParams: ParamDef[] = [
  { id: 'p1', name: 'period', type: 'int', default: 14, min: 2, max: 100 },
  { id: 'p2', name: 'overbought', type: 'int', default: 70, min: 50, max: 95 },
  { id: 'p3', name: 'oversold', type: 'int', default: 30, min: 5, max: 50 },
];

const initialOutputs: OutputState[] = [
  { id: 'o1', code: 'OVERBOUGHT', description: 'RSI above overbought threshold' },
  { id: 'o2', code: 'NEUTRAL', description: 'RSI between oversold and overbought' },
  { id: 'o3', code: 'OVERSOLD', description: 'RSI below oversold threshold' },
];

const initialTriggers: TriggerDef[] = [
  { id: 't1', name: 'RSI Cross Above OB', direction: 'SHORT', type: 'ENTRY', execution: '[C]' },
  { id: 't2', name: 'RSI Cross Below OS', direction: 'LONG', type: 'ENTRY', execution: '[C]' },
];

const execTypeOptions = ['[C]', '[L0]', '[L1]', '[HM]', '[HL]'];
const directionOptions: Array<'LONG' | 'SHORT' | 'BOTH'> = ['LONG', 'SHORT', 'BOTH'];
const triggerTypeOptions: Array<'ENTRY' | 'EXIT'> = ['ENTRY', 'EXIT'];
const categoryOptions = ['Moving Averages', 'Momentum', 'Volume', 'Volatility', 'Trend', 'Custom'];

const mockValidation: ValidationItem[] = [
  { label: 'Syntax Check', status: 'pass', message: 'Indicator and interpreter code parses correctly' },
  { label: 'Output Coverage', status: 'pass', message: 'All 3 output states have defined paths' },
  { label: 'Parameter Types', status: 'pass', message: 'All parameter types and ranges are valid' },
  { label: 'Trigger States', status: 'warn', message: 'Trigger "RSI Cross Below OS" references state transition not fully covered' },
  { label: 'Naming Conventions', status: 'pass', message: 'Pack ID, output codes, and trigger IDs follow conventions' },
];

const mockDetectedOutputs = ['OVERBOUGHT', 'NEUTRAL', 'OVERSOLD'];

/* ========================================================================
   Style Constants
   ======================================================================== */

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

const validationStatusStyles: Record<string, { color: string; icon: string }> = {
  'pass': { color: 'var(--green)', icon: 'OK' },
  'fail': { color: 'var(--red)', icon: 'X' },
  'warn': { color: 'var(--orange)', icon: '!' },
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

/* ========================================================================
   Pack Info Card
   ======================================================================== */

function PackInfoCard({
  packName,
  setPackName,
  category,
  setCategory,
  description,
  setDescription,
  versionName,
  setVersionName,
}: {
  packName: string;
  setPackName: (v: string) => void;
  category: string;
  setCategory: (v: string) => void;
  description: string;
  setDescription: (v: string) => void;
  versionName: string;
  setVersionName: (v: string) => void;
}) {
  return (
    <Card>
      <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Pack Info</h3>
      <div className="space-y-4">
        <div>
          <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>Name</label>
          <input
            className="w-full px-3 py-2 rounded-lg text-sm"
            style={inputStyle}
            value={packName}
            onChange={(e) => setPackName(e.target.value)}
            placeholder="e.g. My RSI Pack"
          />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>Category</label>
            <select
              className="w-full px-3 py-2 rounded-lg text-sm"
              style={selectStyle}
              value={category}
              onChange={(e) => setCategory(e.target.value)}
            >
              {categoryOptions.map((cat) => (
                <option key={cat} value={cat}>{cat}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>Version</label>
            <input
              className="w-full px-3 py-2 rounded-lg text-sm"
              style={inputStyle}
              value={versionName}
              onChange={(e) => setVersionName(e.target.value)}
              placeholder="e.g. v1.0"
            />
          </div>
        </div>
        <div>
          <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>Description</label>
          <textarea
            className="w-full px-3 py-2 rounded-lg text-sm"
            style={{ ...inputStyle, resize: 'vertical' as const }}
            rows={3}
            placeholder="Describe what this pack does..."
            value={description}
            onChange={(e) => setDescription(e.target.value)}
          />
        </div>

        {/* Pack ID preview */}
        <div>
          <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>Pack ID</label>
          <p
            className="text-xs font-mono px-3 py-2 rounded-lg"
            style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}
          >
            {packName ? packName.toLowerCase().replace(/\s+/g, '-') : 'my-custom-pack'}
          </p>
        </div>
      </div>
    </Card>
  );
}

/* ========================================================================
   Indicator Logic Card
   ======================================================================== */

function IndicatorLogicCard({
  params,
  setParams,
  packName,
}: {
  params: ParamDef[];
  setParams: (p: ParamDef[]) => void;
  packName: string;
}) {
  function addParam() {
    setParams([
      ...params,
      {
        id: `p${Date.now()}`,
        name: '',
        type: 'int',
        default: 0,
        min: 0,
        max: 100,
      },
    ]);
  }

  function removeParam(id: string) {
    setParams(params.filter((p) => p.id !== id));
  }

  function updateParam(id: string, field: keyof ParamDef, value: string | number) {
    setParams(
      params.map((p) => (p.id === id ? { ...p, [field]: value } : p))
    );
  }

  const funcName = packName ? packName.toLowerCase().replace(/\s+/g, '_') : 'my_indicator';

  return (
    <Card>
      <h3 className="text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Indicator Logic</h3>
      <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>Define your indicator computation function</p>

      {/* Code editor placeholder */}
      <div
        className="rounded-lg p-4 mb-4"
        style={{
          background: '#0d1117',
          border: '1px solid var(--border)',
          fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',
          minHeight: 200,
        }}
      >
        <p className="text-xs" style={{ color: '#8b949e' }}>
          <span style={{ color: '#ff7b72' }}>def</span>{' '}
          <span style={{ color: '#d2a8ff' }}>compute_indicator</span>
          <span style={{ color: '#c9d1d9' }}>(df, params):</span>
        </p>
        <p className="text-xs mt-1" style={{ color: '#8b949e' }}>
          &nbsp;&nbsp;<span style={{ color: '#8b949e' }}># Compute {packName || 'indicator'} values</span>
        </p>
        <p className="text-xs mt-1" style={{ color: '#c9d1d9' }}>
          &nbsp;&nbsp;<span style={{ color: '#ff7b72' }}>for</span> key <span style={{ color: '#ff7b72' }}>in</span> params:
        </p>
        <p className="text-xs" style={{ color: '#c9d1d9' }}>
          &nbsp;&nbsp;&nbsp;&nbsp;val = params[key]
        </p>
        <p className="text-xs mt-2" style={{ color: '#c9d1d9' }}>
          &nbsp;&nbsp;<span style={{ color: '#8b949e' }}># Calculate indicator column</span>
        </p>
        {params.length > 0 && (
          <>
            <p className="text-xs" style={{ color: '#c9d1d9' }}>
              &nbsp;&nbsp;period = params[<span style={{ color: '#a5d6ff' }}>&apos;{params[0].name || 'period'}&apos;</span>]
            </p>
            <p className="text-xs" style={{ color: '#c9d1d9' }}>
              &nbsp;&nbsp;df[<span style={{ color: '#a5d6ff' }}>&apos;{funcName}&apos;</span>] = df[<span style={{ color: '#a5d6ff' }}>&apos;close&apos;</span>].rolling(period).mean()
            </p>
          </>
        )}
        <p className="text-xs mt-2" style={{ color: '#c9d1d9' }}>
          &nbsp;&nbsp;<span style={{ color: '#ff7b72' }}>return</span> df
        </p>
      </div>

      {/* Parameter Schema Builder */}
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Parameter Schema</h4>
        <button
          onClick={addParam}
          className="px-3 py-1 rounded text-xs font-medium"
          style={btnPrimary}
        >
          + Add Parameter
        </button>
      </div>

      <div className="space-y-2">
        {params.map((param) => (
          <div
            key={param.id}
            className="rounded-lg p-3"
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
          >
            <div className="grid grid-cols-6 gap-2 items-end">
              <div className="col-span-2">
                <label className="text-xs mb-0.5 block" style={{ color: 'var(--text-muted)' }}>Name</label>
                <input
                  className="w-full px-2 py-1.5 rounded text-xs font-mono"
                  style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                  placeholder="param_name"
                  value={param.name}
                  onChange={(e) => updateParam(param.id, 'name', e.target.value)}
                />
              </div>
              <div>
                <label className="text-xs mb-0.5 block" style={{ color: 'var(--text-muted)' }}>Type</label>
                <select
                  className="w-full px-2 py-1.5 rounded text-xs"
                  style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                  value={param.type}
                  onChange={(e) => updateParam(param.id, 'type', e.target.value)}
                >
                  <option value="int">int</option>
                  <option value="float">float</option>
                </select>
              </div>
              <div>
                <label className="text-xs mb-0.5 block" style={{ color: 'var(--text-muted)' }}>Default</label>
                <input
                  type="number"
                  className="w-full px-2 py-1.5 rounded text-xs font-mono"
                  style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                  value={param.default}
                  onChange={(e) => updateParam(param.id, 'default', Number(e.target.value))}
                />
              </div>
              <div>
                <label className="text-xs mb-0.5 block" style={{ color: 'var(--text-muted)' }}>Min / Max</label>
                <div className="flex gap-1">
                  <input
                    type="number"
                    className="w-full px-1.5 py-1.5 rounded text-xs font-mono"
                    style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                    value={param.min}
                    onChange={(e) => updateParam(param.id, 'min', Number(e.target.value))}
                  />
                  <input
                    type="number"
                    className="w-full px-1.5 py-1.5 rounded text-xs font-mono"
                    style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                    value={param.max}
                    onChange={(e) => updateParam(param.id, 'max', Number(e.target.value))}
                  />
                </div>
              </div>
              <div className="flex justify-end">
                <button
                  onClick={() => removeParam(param.id)}
                  className="w-7 h-7 rounded flex items-center justify-center text-xs"
                  style={{ color: 'var(--red)', background: 'var(--red-muted)' }}
                >
                  x
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {params.length === 0 && (
        <p className="text-xs text-center py-4" style={{ color: 'var(--text-muted)' }}>
          No parameters defined. Click &quot;+ Add Parameter&quot; to add one.
        </p>
      )}
    </Card>
  );
}

/* ========================================================================
   Interpreter Logic Card
   ======================================================================== */

function InterpreterLogicCard({
  outputs,
  setOutputs,
}: {
  outputs: OutputState[];
  setOutputs: (o: OutputState[]) => void;
}) {
  function addOutput() {
    setOutputs([
      ...outputs,
      { id: `o${Date.now()}`, code: '', description: '' },
    ]);
  }

  function removeOutput(id: string) {
    setOutputs(outputs.filter((o) => o.id !== id));
  }

  function updateOutput(id: string, field: keyof OutputState, value: string) {
    setOutputs(
      outputs.map((o) => (o.id === id ? { ...o, [field]: value } : o))
    );
  }

  return (
    <Card>
      <h3 className="text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Interpreter Logic</h3>
      <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>Define state classification rules from indicator values</p>

      {/* Code editor placeholder */}
      <div
        className="rounded-lg p-4 mb-4"
        style={{
          background: '#0d1117',
          border: '1px solid var(--border)',
          fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',
          minHeight: 160,
        }}
      >
        <p className="text-xs" style={{ color: '#8b949e' }}>
          <span style={{ color: '#ff7b72' }}>def</span>{' '}
          <span style={{ color: '#d2a8ff' }}>interpret</span>
          <span style={{ color: '#c9d1d9' }}>(df, indicator_values):</span>
        </p>
        <p className="text-xs mt-1" style={{ color: '#8b949e' }}>
          &nbsp;&nbsp;<span style={{ color: '#8b949e' }}># Classify current state from indicator values</span>
        </p>
        {outputs.map((output, i) => (
          <div key={output.id}>
            <p className="text-xs" style={{ color: '#c9d1d9' }}>
              &nbsp;&nbsp;<span style={{ color: '#ff7b72' }}>{i === 0 ? 'if' : 'elif'}</span> condition_{i + 1}:
            </p>
            <p className="text-xs" style={{ color: '#c9d1d9' }}>
              &nbsp;&nbsp;&nbsp;&nbsp;<span style={{ color: '#ff7b72' }}>return</span>{' '}
              <span style={{ color: '#a5d6ff' }}>&apos;{output.code || `STATE_${i + 1}`}&apos;</span>
            </p>
          </div>
        ))}
        {outputs.length === 0 && (
          <p className="text-xs mt-1" style={{ color: '#c9d1d9' }}>
            &nbsp;&nbsp;<span style={{ color: '#ff7b72' }}>return</span>{' '}
            <span style={{ color: '#a5d6ff' }}>&apos;UNKNOWN&apos;</span>
          </p>
        )}
      </div>

      {/* Output State Definer */}
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Output States</h4>
        <button
          onClick={addOutput}
          className="px-3 py-1 rounded text-xs font-medium"
          style={btnPrimary}
        >
          + Add Output
        </button>
      </div>

      <div className="space-y-2">
        {outputs.map((output) => (
          <div
            key={output.id}
            className="rounded-lg p-3 flex items-center gap-3"
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
          >
            <div className="w-32 flex-shrink-0">
              <label className="text-xs mb-0.5 block" style={{ color: 'var(--text-muted)' }}>Code</label>
              <input
                className="w-full px-2 py-1.5 rounded text-xs font-mono font-semibold"
                style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                placeholder="STATE_CODE"
                value={output.code}
                onChange={(e) => updateOutput(output.id, 'code', e.target.value.toUpperCase())}
              />
            </div>
            <div className="flex-1">
              <label className="text-xs mb-0.5 block" style={{ color: 'var(--text-muted)' }}>Description</label>
              <input
                className="w-full px-2 py-1.5 rounded text-xs"
                style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                placeholder="Describe when this state is active"
                value={output.description}
                onChange={(e) => updateOutput(output.id, 'description', e.target.value)}
              />
            </div>
            <button
              onClick={() => removeOutput(output.id)}
              className="w-7 h-7 rounded flex items-center justify-center text-xs flex-shrink-0 mt-4"
              style={{ color: 'var(--red)', background: 'var(--red-muted)' }}
            >
              x
            </button>
          </div>
        ))}
      </div>

      {outputs.length === 0 && (
        <p className="text-xs text-center py-4" style={{ color: 'var(--text-muted)' }}>
          No output states defined. Click &quot;+ Add Output&quot; to add one.
        </p>
      )}
    </Card>
  );
}

/* ========================================================================
   Trigger Definitions Card
   ======================================================================== */

function TriggerDefinitionsCard({
  triggers,
  setTriggers,
}: {
  triggers: TriggerDef[];
  setTriggers: (t: TriggerDef[]) => void;
}) {
  function addTrigger() {
    setTriggers([
      ...triggers,
      {
        id: `t${Date.now()}`,
        name: '',
        direction: 'LONG',
        type: 'ENTRY',
        execution: '[C]',
      },
    ]);
  }

  function removeTrigger(id: string) {
    setTriggers(triggers.filter((t) => t.id !== id));
  }

  function updateTrigger(id: string, field: keyof TriggerDef, value: string) {
    setTriggers(
      triggers.map((t) => (t.id === id ? { ...t, [field]: value } : t))
    );
  }

  return (
    <Card>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Trigger Definitions</h3>
          <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
            Define state transition triggers with direction and execution type
          </p>
        </div>
        <button
          onClick={addTrigger}
          className="px-3 py-1.5 rounded text-xs font-medium"
          style={btnPrimary}
        >
          + Add Trigger
        </button>
      </div>

      <div className="space-y-3">
        {triggers.map((trigger) => (
          <div
            key={trigger.id}
            className="rounded-lg p-3"
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
          >
            {/* Row 1: Name + Delete */}
            <div className="flex items-center gap-2 mb-3">
              <input
                className="px-2 py-1.5 rounded text-sm flex-1"
                style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                placeholder="Trigger name..."
                value={trigger.name}
                onChange={(e) => updateTrigger(trigger.id, 'name', e.target.value)}
              />
              <button
                onClick={() => removeTrigger(trigger.id)}
                className="w-7 h-7 rounded flex items-center justify-center text-xs"
                style={{ color: 'var(--red)', background: 'var(--red-muted)' }}
              >
                x
              </button>
            </div>

            {/* Row 2: Direction, Type, Execution */}
            <div className="flex items-center gap-3">
              <div className="flex-1">
                <label className="text-xs mb-0.5 block" style={{ color: 'var(--text-muted)' }}>Direction</label>
                <select
                  className="w-full px-2 py-1.5 rounded text-xs"
                  style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                  value={trigger.direction}
                  onChange={(e) => updateTrigger(trigger.id, 'direction', e.target.value)}
                >
                  {directionOptions.map((d) => (
                    <option key={d} value={d}>{d}</option>
                  ))}
                </select>
              </div>
              <div className="flex-1">
                <label className="text-xs mb-0.5 block" style={{ color: 'var(--text-muted)' }}>Type</label>
                <select
                  className="w-full px-2 py-1.5 rounded text-xs"
                  style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                  value={trigger.type}
                  onChange={(e) => updateTrigger(trigger.id, 'type', e.target.value)}
                >
                  {triggerTypeOptions.map((t) => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>
              </div>
              <div className="flex-1">
                <label className="text-xs mb-0.5 block" style={{ color: 'var(--text-muted)' }}>Execution</label>
                <select
                  className="w-full px-2 py-1.5 rounded text-xs"
                  style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                  value={trigger.execution}
                  onChange={(e) => updateTrigger(trigger.id, 'execution', e.target.value)}
                >
                  {execTypeOptions.map((et) => (
                    <option key={et} value={et}>{et}</option>
                  ))}
                </select>
              </div>
            </div>

            {/* Preview badges */}
            {trigger.name && (
              <div className="flex items-center gap-2 mt-3 pt-2 border-t" style={{ borderColor: 'var(--border)' }}>
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Preview:</span>
                <span className="text-xs font-medium" style={{ color: 'var(--text-primary)' }}>{trigger.name}</span>
                <DirectionBadge dir={trigger.direction} />
                <TypeBadge type={trigger.type} />
                <ExecBadge exec={trigger.execution} />
              </div>
            )}
          </div>
        ))}
      </div>

      {triggers.length === 0 && (
        <p className="text-xs text-center py-4" style={{ color: 'var(--text-muted)' }}>
          No triggers defined. Click &quot;+ Add Trigger&quot; to add one.
        </p>
      )}
    </Card>
  );
}

/* ========================================================================
   Live Preview Card
   ======================================================================== */

function LivePreviewCard({ packName }: { packName: string }) {
  const [symbol, setSymbol] = useState('NVDA');
  const [timeframe, setTimeframe] = useState('5Min');
  const [session, setSession] = useState('RTH Only');

  return (
    <Card>
      <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Live Preview</h3>
      <div className="flex gap-2 mb-3">
        <select
          className="px-3 py-1.5 rounded-lg text-xs"
          style={selectStyle}
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
        >
          {['NVDA', 'SPY', 'AAPL', 'TSLA', 'MSFT', 'AMD'].map((s) => (
            <option key={s}>{s}</option>
          ))}
        </select>
        <select
          className="px-3 py-1.5 rounded-lg text-xs"
          style={selectStyle}
          value={timeframe}
          onChange={(e) => setTimeframe(e.target.value)}
        >
          {['1Min', '5Min', '15Min', '1H', '4H', '1D'].map((tf) => (
            <option key={tf}>{tf}</option>
          ))}
        </select>
        <select
          className="px-3 py-1.5 rounded-lg text-xs"
          style={selectStyle}
          value={session}
          onChange={(e) => setSession(e.target.value)}
        >
          <option>RTH Only</option>
          <option>Extended Hours</option>
          <option>24/7</option>
        </select>
      </div>
      <ChartPlaceholder label={`${symbol} ${timeframe} price chart with ${packName || 'indicator'} overlay`} height={300} />
    </Card>
  );
}

/* ========================================================================
   Detected Outputs Card
   ======================================================================== */

function DetectedOutputsCard({ outputs }: { outputs: OutputState[] }) {
  return (
    <Card>
      <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Detected Outputs</h3>
      <div className="space-y-2">
        {(outputs.length > 0 ? outputs : mockDetectedOutputs.map((code) => ({ id: code, code, description: '' }))).map((output) => (
          <div
            key={output.id}
            className="flex items-center justify-between px-3 py-2 rounded-lg"
            style={{ background: 'var(--bg-input)' }}
          >
            <span className="text-sm font-mono" style={{ color: 'var(--text-primary)' }}>{output.code || 'UNNAMED'}</span>
            <span className="text-xs" style={{ color: 'var(--green)' }}>detected</span>
          </div>
        ))}
      </div>
    </Card>
  );
}

/* ========================================================================
   Test Results Card
   ======================================================================== */

function TestResultsCard({ triggers }: { triggers: TriggerDef[] }) {
  return (
    <Card>
      <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Test Results</h3>
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)' }}>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Trigger Fires</p>
          <p className="text-lg font-semibold">47</p>
        </div>
        <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)' }}>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Signal Accuracy</p>
          <p className="text-lg font-semibold">68.3%</p>
        </div>
        <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)' }}>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Avg Bars Between</p>
          <p className="text-lg font-semibold">51</p>
        </div>
        <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)' }}>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>State Coverage</p>
          <p className="text-lg font-semibold">3/3</p>
        </div>
      </div>

      {/* Per-trigger breakdown */}
      <div className="border-t pt-3" style={{ borderColor: 'var(--border)' }}>
        <p className="text-xs font-medium mb-2" style={{ color: 'var(--text-muted)' }}>Per-Trigger Breakdown</p>
        {triggers.filter((t) => t.name).map((trigger) => (
          <div
            key={trigger.id}
            className="flex items-center justify-between py-2 border-b"
            style={{ borderColor: 'var(--border)' }}
          >
            <div className="flex items-center gap-2">
              <span className="text-sm" style={{ color: 'var(--text-primary)' }}>{trigger.name}</span>
              <ExecBadge exec={trigger.execution} />
            </div>
            <div className="flex items-center gap-4">
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                {Math.floor(20 + trigger.name.length * 2)} fires
              </span>
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                {(1 + trigger.name.length * 0.1).toFixed(1)}%
              </span>
            </div>
          </div>
        ))}
        {triggers.filter((t) => t.name).length === 0 && (
          <p className="text-xs py-2" style={{ color: 'var(--text-muted)' }}>
            Define triggers with names to see breakdown
          </p>
        )}
      </div>
    </Card>
  );
}

/* ========================================================================
   Validation Panel Card
   ======================================================================== */

function ValidationPanelCard() {
  return (
    <Card>
      <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Validation</h3>
      <div className="space-y-2">
        {mockValidation.map((item) => {
          const vs = validationStatusStyles[item.status];
          return (
            <div
              key={item.label}
              className="flex items-start gap-3 px-3 py-2.5 rounded-lg"
              style={{ background: 'var(--bg-input)' }}
            >
              <span
                className="text-xs font-mono font-bold w-5 h-5 rounded flex items-center justify-center flex-shrink-0 mt-0.5"
                style={{
                  color: vs.color,
                  background: item.status === 'pass' ? 'var(--green-muted)' :
                              item.status === 'fail' ? 'var(--red-muted)' : 'var(--orange-muted)',
                }}
              >
                {vs.icon}
              </span>
              <div className="flex-1">
                <p className="text-xs font-medium" style={{ color: 'var(--text-primary)' }}>{item.label}</p>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{item.message}</p>
              </div>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

/* ========================================================================
   Main Component
   ======================================================================== */

export default function PackBuilderV2() {
  const [packName, setPackName] = useState('My RSI Pack');
  const [category, setCategory] = useState('Momentum');
  const [description, setDescription] = useState('Custom RSI implementation with configurable overbought/oversold zones');
  const [versionName, setVersionName] = useState('v1.0');
  const [params, setParams] = useState<ParamDef[]>(initialParams);
  const [outputs, setOutputs] = useState<OutputState[]>(initialOutputs);
  const [triggers, setTriggers] = useState<TriggerDef[]>(initialTriggers);

  return (
    <div>
      <PageHeader
        title="Pack Builder"
        subtitle="Build custom indicator packs with your own logic"
        actions={
          <div className="flex gap-2">
            <button
              className="px-4 py-2 rounded-lg text-sm"
              style={btnSecondary}
            >
              Save Draft
            </button>
            <button
              className="px-4 py-2 rounded-lg text-sm font-medium"
              style={btnPrimary}
            >
              Publish Pack
            </button>
          </div>
        }
      />

      <div className="grid grid-cols-2 gap-6">
        {/* Left column -- Config */}
        <div className="space-y-6">
          <PackInfoCard
            packName={packName}
            setPackName={setPackName}
            category={category}
            setCategory={setCategory}
            description={description}
            setDescription={setDescription}
            versionName={versionName}
            setVersionName={setVersionName}
          />

          <IndicatorLogicCard
            params={params}
            setParams={setParams}
            packName={packName}
          />

          <InterpreterLogicCard
            outputs={outputs}
            setOutputs={setOutputs}
          />

          <TriggerDefinitionsCard
            triggers={triggers}
            setTriggers={setTriggers}
          />
        </div>

        {/* Right column -- Preview & Validation */}
        <div className="space-y-6">
          <LivePreviewCard packName={packName} />

          <DetectedOutputsCard outputs={outputs} />

          <TestResultsCard triggers={triggers} />

          <ValidationPanelCard />
        </div>
      </div>

      {/* Bottom save bar */}
      <div
        className="mt-8 pt-6 border-t flex items-center justify-between"
        style={{ borderColor: 'var(--border)' }}
      >
        <div className="flex items-center gap-3">
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {params.length} params &middot; {outputs.length} outputs &middot; {triggers.length} triggers
          </span>
          <div className="border-l h-4" style={{ borderColor: 'var(--border)' }} />
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
            Pack ID: {packName ? packName.toLowerCase().replace(/\s+/g, '-') : 'unnamed'}
          </span>
        </div>
        <div className="flex gap-3">
          <button
            className="px-5 py-2.5 rounded-lg text-sm"
            style={btnSecondary}
          >
            Save Draft
          </button>
          <button
            className="px-5 py-2.5 rounded-lg text-sm font-medium"
            style={btnPrimary}
          >
            Publish Pack
          </button>
        </div>
      </div>
    </div>
  );
}
