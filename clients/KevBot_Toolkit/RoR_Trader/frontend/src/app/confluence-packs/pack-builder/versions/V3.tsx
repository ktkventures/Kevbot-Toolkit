'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

/* ---------- Types ---------- */
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

/* ---------- Constants ---------- */
const execTypeOptions = ['[C]', '[L0]', '[L1]', '[HM]', '[HL]'];
const directionOptions: Array<'LONG' | 'SHORT' | 'BOTH'> = ['LONG', 'SHORT', 'BOTH'];
const triggerTypeOptions: Array<'ENTRY' | 'EXIT'> = ['ENTRY', 'EXIT'];
const categoryOptions = ['Moving Averages', 'Momentum', 'Volume', 'Volatility', 'Trend', 'Custom'];

const execTypeBadgeStyles: Record<string, { color: string; bg: string }> = {
  '[C]': { color: 'var(--green)', bg: 'var(--green-muted)' },
  '[L0]': { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  '[L1]': { color: 'var(--accent)', bg: 'var(--accent-muted)' },
  '[HM]': { color: 'var(--orange)', bg: 'var(--orange-muted)' },
  '[HL]': { color: 'var(--red)', bg: 'var(--red-muted)' },
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

/* ---------- Initial Data ---------- */
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

const mockValidation: ValidationItem[] = [
  { label: 'Parameter Types', status: 'pass', message: 'All parameter types and ranges are valid' },
  { label: 'Output Coverage', status: 'pass', message: 'All 3 output states defined' },
  { label: 'Trigger States', status: 'warn', message: '1 trigger references a transition not fully mapped' },
  { label: 'Naming Conventions', status: 'pass', message: 'Pack ID and output codes follow conventions' },
];

/* ---------- Collapsible Section ---------- */
function StepSection({
  step,
  title,
  subtitle,
  expanded,
  onToggle,
  valid,
  children,
}: {
  step: number;
  title: string;
  subtitle: string;
  expanded: boolean;
  onToggle: () => void;
  valid: boolean;
  children: React.ReactNode;
}) {
  return (
    <div
      className="rounded-xl border overflow-hidden"
      style={{
        background: 'var(--bg-card)',
        borderColor: expanded ? 'var(--accent)' : 'var(--border)',
        boxShadow: 'var(--card-shadow)',
        backdropFilter: 'var(--card-backdrop)',
        WebkitBackdropFilter: 'var(--card-backdrop)',
      }}
    >
      <div
        className="flex items-center gap-3 px-4 py-3 cursor-pointer select-none"
        onClick={onToggle}
      >
        {/* Step number */}
        <div
          className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0"
          style={{
            background: valid ? 'var(--accent)' : 'var(--bg-input)',
            color: valid ? 'white' : 'var(--text-muted)',
            border: valid ? 'none' : '1px solid var(--border)',
          }}
        >
          {step}
        </div>

        {/* Title + subtitle */}
        <div className="flex-1 min-w-0">
          <span className="font-semibold text-sm" style={{ color: 'var(--text-primary)' }}>
            {title}
          </span>
          <span className="text-xs ml-2" style={{ color: 'var(--text-muted)' }}>
            {subtitle}
          </span>
        </div>

        {/* Valid badge */}
        {valid && (
          <span className="text-xs px-2 py-0.5 rounded flex-shrink-0" style={{ color: 'var(--green)', background: 'var(--green-muted)' }}>
            OK
          </span>
        )}

        {/* Chevron */}
        <span
          className="text-sm flex-shrink-0 transition-transform"
          style={{
            color: 'var(--text-muted)',
            transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
          }}
        >
          &#9662;
        </span>
      </div>

      {expanded && (
        <div className="px-4 pb-4 border-t" style={{ borderColor: 'var(--border)' }}>
          <div className="pt-4">
            {children}
          </div>
        </div>
      )}
    </div>
  );
}

/* ---------- Main Component ---------- */
export default function PackBuilderV3() {
  // Step 1: Info
  const [packName, setPackName] = useState('My Custom Pack');
  const [category, setCategory] = useState('Momentum');
  const [description, setDescription] = useState('');

  // Step 2: Parameters
  const [params, setParams] = useState<ParamDef[]>(initialParams);

  // Step 3: Outputs
  const [outputs, setOutputs] = useState<OutputState[]>(initialOutputs);

  // Step 4: Triggers
  const [triggers, setTriggers] = useState<TriggerDef[]>(initialTriggers);

  // UI state
  const [expandedStep, setExpandedStep] = useState<number>(1);

  // Param CRUD
  function addParam() {
    setParams((prev) => [
      ...prev,
      { id: `p${Date.now()}`, name: '', type: 'int', default: 0, min: 0, max: 100 },
    ]);
  }

  function removeParam(id: string) {
    setParams((prev) => prev.filter((p) => p.id !== id));
  }

  function updateParam(id: string, field: keyof ParamDef, value: string | number) {
    setParams((prev) => prev.map((p) => p.id === id ? { ...p, [field]: value } : p));
  }

  // Output CRUD
  function addOutput() {
    setOutputs((prev) => [
      ...prev,
      { id: `o${Date.now()}`, code: '', description: '' },
    ]);
  }

  function removeOutput(id: string) {
    setOutputs((prev) => prev.filter((o) => o.id !== id));
  }

  function updateOutput(id: string, field: keyof OutputState, value: string) {
    setOutputs((prev) => prev.map((o) => o.id === id ? { ...o, [field]: value } : o));
  }

  // Trigger CRUD
  function addTrigger() {
    setTriggers((prev) => [
      ...prev,
      { id: `t${Date.now()}`, name: '', direction: 'LONG', type: 'ENTRY', execution: '[C]' },
    ]);
  }

  function removeTrigger(id: string) {
    setTriggers((prev) => prev.filter((t) => t.id !== id));
  }

  function updateTrigger(id: string, field: keyof TriggerDef, value: string) {
    setTriggers((prev) => prev.map((t) => t.id === id ? { ...t, [field]: value } : t));
  }

  // Validation
  const infoValid = packName.trim().length > 0;
  const paramsValid = params.length > 0 && params.every((p) => p.name.trim().length > 0);
  const outputsValid = outputs.length > 0 && outputs.every((o) => o.code.trim().length > 0);
  const triggersValid = triggers.every((t) => t.name.trim().length > 0);
  const allValid = infoValid && paramsValid && outputsValid && triggersValid;

  const inputStyle: React.CSSProperties = {
    background: 'var(--bg-input)',
    border: '1px solid var(--border)',
    color: 'var(--text-primary)',
  };

  return (
    <div>
      <PageHeader
        title="Pack Builder"
        subtitle="Build custom indicator packs step by step"
      />

      <div className="space-y-2">
        {/* Step 1: Info */}
        <StepSection
          step={1}
          title="Pack Info"
          subtitle="Name, category, and description"
          expanded={expandedStep === 1}
          onToggle={() => setExpandedStep(expandedStep === 1 ? 0 : 1)}
          valid={infoValid}
        >
          <div className="space-y-4 max-w-lg">
            <div>
              <label className="text-xs font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>Name</label>
              <input
                className="w-full px-3 py-2 rounded-lg text-sm"
                style={inputStyle}
                value={packName}
                onChange={(e) => setPackName(e.target.value)}
              />
              {!packName.trim() && (
                <p className="text-xs mt-1" style={{ color: 'var(--red)' }}>Name is required</p>
              )}
            </div>
            <div>
              <label className="text-xs font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>Category</label>
              <select
                className="w-full px-3 py-2 rounded-lg text-sm"
                style={inputStyle}
                value={category}
                onChange={(e) => setCategory(e.target.value)}
              >
                {categoryOptions.map((cat) => (
                  <option key={cat} value={cat}>{cat}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-xs font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>Description</label>
              <textarea
                className="w-full px-3 py-2 rounded-lg text-sm"
                style={{ ...inputStyle, resize: 'vertical' as const }}
                rows={2}
                placeholder="Describe what this pack does..."
                value={description}
                onChange={(e) => setDescription(e.target.value)}
              />
            </div>
          </div>
        </StepSection>

        {/* Step 2: Parameters */}
        <StepSection
          step={2}
          title="Parameters"
          subtitle={`${params.length} defined`}
          expanded={expandedStep === 2}
          onToggle={() => setExpandedStep(expandedStep === 2 ? 0 : 2)}
          valid={paramsValid}
        >
          <div className="space-y-3">
            {params.map((param) => (
              <div
                key={param.id}
                className="rounded-lg p-3"
                style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}
              >
                <div className="flex items-center gap-2 mb-2">
                  <input
                    className="px-2 py-1 rounded text-sm flex-1"
                    style={inputStyle}
                    placeholder="Parameter name..."
                    value={param.name}
                    onChange={(e) => updateParam(param.id, 'name', e.target.value)}
                  />
                  <select
                    className="px-2 py-1 rounded text-sm w-20"
                    style={inputStyle}
                    value={param.type}
                    onChange={(e) => updateParam(param.id, 'type', e.target.value)}
                  >
                    <option value="int">Int</option>
                    <option value="float">Float</option>
                  </select>
                  <button
                    onClick={() => removeParam(param.id)}
                    className="w-7 h-7 rounded flex items-center justify-center text-xs flex-shrink-0"
                    style={{ color: 'var(--red)', background: 'var(--red-muted)' }}
                  >
                    x
                  </button>
                </div>
                <div className="flex items-center gap-2">
                  <div className="flex-1">
                    <label className="text-xs block mb-0.5" style={{ color: 'var(--text-muted)' }}>Default</label>
                    <input
                      type="number"
                      className="w-full px-2 py-1 rounded text-xs"
                      style={inputStyle}
                      value={param.default}
                      onChange={(e) => updateParam(param.id, 'default', parseFloat(e.target.value) || 0)}
                    />
                  </div>
                  <div className="flex-1">
                    <label className="text-xs block mb-0.5" style={{ color: 'var(--text-muted)' }}>Min</label>
                    <input
                      type="number"
                      className="w-full px-2 py-1 rounded text-xs"
                      style={inputStyle}
                      value={param.min}
                      onChange={(e) => updateParam(param.id, 'min', parseFloat(e.target.value) || 0)}
                    />
                  </div>
                  <div className="flex-1">
                    <label className="text-xs block mb-0.5" style={{ color: 'var(--text-muted)' }}>Max</label>
                    <input
                      type="number"
                      className="w-full px-2 py-1 rounded text-xs"
                      style={inputStyle}
                      value={param.max}
                      onChange={(e) => updateParam(param.id, 'max', parseFloat(e.target.value) || 100)}
                    />
                  </div>
                </div>
                {!param.name.trim() && (
                  <p className="text-xs mt-1" style={{ color: 'var(--red)' }}>Parameter name is required</p>
                )}
              </div>
            ))}
            <button
              onClick={addParam}
              className="px-3 py-1.5 rounded text-xs font-medium"
              style={{ background: 'var(--accent)', color: 'white' }}
            >
              + Add Parameter
            </button>
          </div>
        </StepSection>

        {/* Step 3: Outputs */}
        <StepSection
          step={3}
          title="Outputs"
          subtitle={`${outputs.length} states`}
          expanded={expandedStep === 3}
          onToggle={() => setExpandedStep(expandedStep === 3 ? 0 : 3)}
          valid={outputsValid}
        >
          <div className="space-y-3">
            {outputs.map((output) => (
              <div
                key={output.id}
                className="flex items-center gap-2"
              >
                <input
                  className="px-2 py-1.5 rounded text-sm font-mono w-40"
                  style={inputStyle}
                  placeholder="CODE"
                  value={output.code}
                  onChange={(e) => updateOutput(output.id, 'code', e.target.value.toUpperCase())}
                />
                <input
                  className="px-2 py-1.5 rounded text-sm flex-1"
                  style={inputStyle}
                  placeholder="Description..."
                  value={output.description}
                  onChange={(e) => updateOutput(output.id, 'description', e.target.value)}
                />
                <button
                  onClick={() => removeOutput(output.id)}
                  className="w-7 h-7 rounded flex items-center justify-center text-xs flex-shrink-0"
                  style={{ color: 'var(--red)', background: 'var(--red-muted)' }}
                >
                  x
                </button>
              </div>
            ))}
            {outputs.length === 0 && (
              <p className="text-xs" style={{ color: 'var(--red)' }}>At least one output state is required</p>
            )}
            <button
              onClick={addOutput}
              className="px-3 py-1.5 rounded text-xs font-medium"
              style={{ background: 'var(--accent)', color: 'white' }}
            >
              + Add Output
            </button>
          </div>
        </StepSection>

        {/* Step 4: Triggers */}
        <StepSection
          step={4}
          title="Triggers"
          subtitle={`${triggers.length} defined`}
          expanded={expandedStep === 4}
          onToggle={() => setExpandedStep(expandedStep === 4 ? 0 : 4)}
          valid={triggersValid}
        >
          <div className="space-y-3">
            {triggers.map((trigger) => (
              <div
                key={trigger.id}
                className="rounded-lg p-3"
                style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}
              >
                <div className="flex items-center gap-2 mb-2">
                  <input
                    className="px-2 py-1 rounded text-sm flex-1"
                    style={inputStyle}
                    placeholder="Trigger name..."
                    value={trigger.name}
                    onChange={(e) => updateTrigger(trigger.id, 'name', e.target.value)}
                  />
                  <button
                    onClick={() => removeTrigger(trigger.id)}
                    className="w-7 h-7 rounded flex items-center justify-center text-xs flex-shrink-0"
                    style={{ color: 'var(--red)', background: 'var(--red-muted)' }}
                  >
                    x
                  </button>
                </div>
                <div className="flex items-center gap-2">
                  <div>
                    <label className="text-xs block mb-0.5" style={{ color: 'var(--text-muted)' }}>Direction</label>
                    <select
                      className="px-2 py-1 rounded text-xs"
                      style={inputStyle}
                      value={trigger.direction}
                      onChange={(e) => updateTrigger(trigger.id, 'direction', e.target.value)}
                    >
                      {directionOptions.map((d) => (
                        <option key={d} value={d}>{d}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="text-xs block mb-0.5" style={{ color: 'var(--text-muted)' }}>Type</label>
                    <select
                      className="px-2 py-1 rounded text-xs"
                      style={inputStyle}
                      value={trigger.type}
                      onChange={(e) => updateTrigger(trigger.id, 'type', e.target.value)}
                    >
                      {triggerTypeOptions.map((t) => (
                        <option key={t} value={t}>{t}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="text-xs block mb-0.5" style={{ color: 'var(--text-muted)' }}>Execution</label>
                    <select
                      className="px-2 py-1 rounded text-xs"
                      style={inputStyle}
                      value={trigger.execution}
                      onChange={(e) => updateTrigger(trigger.id, 'execution', e.target.value)}
                    >
                      {execTypeOptions.map((et) => (
                        <option key={et} value={et}>{et}</option>
                      ))}
                    </select>
                  </div>
                  {/* Live badges */}
                  <div className="flex gap-1 ml-auto mt-3">
                    {(() => {
                      const ds = directionColors[trigger.direction] || { color: 'var(--text-muted)', bg: 'var(--bg-input)' };
                      return (
                        <span className="text-xs px-1.5 py-0.5 rounded" style={{ color: ds.color, background: ds.bg }}>
                          {trigger.direction}
                        </span>
                      );
                    })()}
                    {(() => {
                      const ts = triggerTypeColors[trigger.type] || { color: 'var(--text-muted)', bg: 'var(--bg-input)' };
                      return (
                        <span className="text-xs px-1.5 py-0.5 rounded" style={{ color: ts.color, background: ts.bg }}>
                          {trigger.type}
                        </span>
                      );
                    })()}
                    {(() => {
                      const es = execTypeBadgeStyles[trigger.execution] || { color: 'var(--text-muted)', bg: 'var(--bg-input)' };
                      return (
                        <span className="text-xs font-mono px-1.5 py-0.5 rounded" style={{ color: es.color, background: es.bg }}>
                          {trigger.execution}
                        </span>
                      );
                    })()}
                  </div>
                </div>
                {!trigger.name.trim() && (
                  <p className="text-xs mt-1" style={{ color: 'var(--red)' }}>Trigger name is required</p>
                )}
              </div>
            ))}
            <button
              onClick={addTrigger}
              className="px-3 py-1.5 rounded text-xs font-medium"
              style={{ background: 'var(--accent)', color: 'white' }}
            >
              + Add Trigger
            </button>
          </div>
        </StepSection>

        {/* Step 5: Preview / Validation */}
        <StepSection
          step={5}
          title="Preview & Validation"
          subtitle={allValid ? 'Ready to save' : 'Fix issues above'}
          expanded={expandedStep === 5}
          onToggle={() => setExpandedStep(expandedStep === 5 ? 0 : 5)}
          valid={allValid}
        >
          <div className="space-y-4">
            {/* Pack summary */}
            <div
              className="rounded-lg p-4"
              style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}
            >
              <h4 className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--text-muted)' }}>
                Pack Summary
              </h4>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Name</p>
                  <p className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>{packName || '(untitled)'}</p>
                </div>
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Category</p>
                  <p className="text-sm" style={{ color: 'var(--text-primary)' }}>{category}</p>
                </div>
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Parameters</p>
                  <p className="text-sm" style={{ color: 'var(--text-primary)' }}>{params.length}</p>
                </div>
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Outputs</p>
                  <div className="flex flex-wrap gap-1 mt-0.5">
                    {outputs.map((o) => (
                      <span key={o.id} className="text-xs font-mono px-1.5 py-0.5 rounded" style={{ background: 'var(--bg-input)', color: 'var(--text-secondary)' }}>
                        {o.code || '?'}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="col-span-2">
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Triggers</p>
                  <div className="space-y-1 mt-0.5">
                    {triggers.map((t) => {
                      const ds = directionColors[t.direction] || { color: 'var(--text-muted)', bg: 'var(--bg-input)' };
                      const es = execTypeBadgeStyles[t.execution] || { color: 'var(--text-muted)', bg: 'var(--bg-input)' };
                      return (
                        <div key={t.id} className="flex items-center gap-2 text-xs">
                          <span className="font-mono px-1 py-0.5 rounded" style={{ color: es.color, background: es.bg }}>{t.execution}</span>
                          <span style={{ color: 'var(--text-primary)' }}>{t.name || '(unnamed)'}</span>
                          <span className="px-1 py-0.5 rounded" style={{ color: ds.color, background: ds.bg, fontSize: '10px' }}>{t.direction}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>

            {/* Validation */}
            <div
              className="rounded-lg p-4"
              style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)' }}
            >
              <h4 className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--text-muted)' }}>
                Validation
              </h4>
              <div className="space-y-2">
                {mockValidation.map((item) => {
                  const vs = validationStatusStyles[item.status];
                  return (
                    <div key={item.label} className="flex items-center gap-3">
                      <span
                        className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0"
                        style={{ color: vs.color, background: item.status === 'pass' ? 'var(--green-muted)' : item.status === 'warn' ? 'var(--orange-muted)' : 'var(--red-muted)' }}
                      >
                        {vs.icon}
                      </span>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>{item.label}</p>
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{item.message}</p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </StepSection>
      </div>

      {/* Save button */}
      <div className="mt-6 flex gap-3">
        <button
          className="px-5 py-2.5 rounded-lg text-sm font-medium"
          style={{
            background: allValid ? 'var(--accent)' : 'var(--bg-input)',
            color: allValid ? 'white' : 'var(--text-muted)',
            cursor: allValid ? 'pointer' : 'not-allowed',
          }}
          disabled={!allValid}
        >
          Save Pack
        </button>
        <button
          className="px-5 py-2.5 rounded-lg text-sm"
          style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
        >
          Save Draft
        </button>
      </div>
    </div>
  );
}
