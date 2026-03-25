'use client';

import { useState, useMemo } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import ChartPlaceholder from '@/components/ChartPlaceholder';

/* ========================================================================= */
/* CONSTANTS                                                                   */
/* ========================================================================= */

const EXEC_BADGE_COLOR = '#2196F3';
const FIDELITY_BADGE_COLOR = '#26C6DA';

const TF_CATEGORIES = ['Momentum', 'Trend', 'Volume', 'Volatility', 'Mean Reversion', 'Custom'];
const GEN_CATEGORIES = ['Time', 'Session', 'Calendar', 'Custom'];
const DISPLAY_TYPES = ['overlay', 'oscillator', 'hidden'] as const;
const EXEC_TYPES = ['[C]', '[L]', '[LC]', '[CC]'] as const;
const SENTIMENTS = ['bullish', 'bearish', 'neutral'] as const;

/* ========================================================================= */
/* TYPES                                                                       */
/* ========================================================================= */

type PackType = 'tf_confluence' | 'general';

interface ParamDef {
  id: string;
  name: string;
  label: string;
  type: 'int' | 'float' | 'bool' | 'select';
  defaultVal: string;
  min: string;
  max: string;
}

interface OutputDef {
  id: string;
  code: string;
  description: string;
}

interface TriggerDef {
  id: string;
  name: string;
  base: string;
  sentiment: 'bullish' | 'bearish' | 'neutral';
  execTypes: string[];
  fidelity: string;
  fromState: string;
  toState: string;
}

interface ValidationItem {
  id: string;
  label: string;
  category: 'schema' | 'safety' | 'functions' | 'execution' | 'backtest';
  status: 'pass' | 'fail' | 'warn' | 'pending';
  message: string;
}

/* ========================================================================= */
/* HELPERS                                                                     */
/* ========================================================================= */

function generateId(): string {
  return 'id-' + Math.random().toString(36).substring(2, 9);
}

function toSlug(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '').substring(0, 40);
}

function ExecBadge({ tag }: { tag: string }) {
  return (
    <span className="text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded-full"
      style={{ color: EXEC_BADGE_COLOR, background: EXEC_BADGE_COLOR + '20' }}>
      {tag}
    </span>
  );
}

function FidelityBadge({ tag }: { tag: string }) {
  return (
    <span className="text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded-full"
      style={{ color: FIDELITY_BADGE_COLOR, background: FIDELITY_BADGE_COLOR + '20' }}>
      {tag}
    </span>
  );
}

function SentimentBadge({ sentiment }: { sentiment: string }) {
  const colors: Record<string, { color: string; bg: string }> = {
    bullish: { color: 'var(--green)', bg: 'var(--green-muted)' },
    bearish: { color: 'var(--red)', bg: 'var(--red-muted)' },
    neutral: { color: 'var(--text-muted)', bg: 'var(--bg-input)' },
  };
  const c = colors[sentiment] || colors.neutral;
  return (
    <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded-full capitalize" style={{ color: c.color, background: c.bg }}>
      {sentiment}
    </span>
  );
}

function StatusIcon({ status }: { status: string }) {
  if (status === 'pass') return <span style={{ color: 'var(--green)' }}>✓</span>;
  if (status === 'fail') return <span style={{ color: 'var(--red)' }}>✗</span>;
  if (status === 'warn') return <span style={{ color: 'var(--orange)' }}>⚠</span>;
  return <span style={{ color: 'var(--text-muted)' }}>○</span>;
}

/* ========================================================================= */
/* STYLES                                                                      */
/* ========================================================================= */

const inputStyle: React.CSSProperties = {
  background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)',
  padding: '6px 10px', borderRadius: '8px', fontSize: '0.8rem', width: '100%',
};

const btnPrimary: React.CSSProperties = {
  background: 'var(--accent)', color: 'white', border: 'none',
  padding: '8px 16px', borderRadius: '8px', fontSize: '0.875rem', cursor: 'pointer', fontWeight: 600,
};

const btnSecondary: React.CSSProperties = {
  background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)',
  padding: '6px 14px', borderRadius: '8px', fontSize: '0.875rem', cursor: 'pointer',
};

/* ========================================================================= */
/* MOCK VALIDATION                                                             */
/* ========================================================================= */

const mockValidation: ValidationItem[] = [
  { id: 'v1', label: 'Required fields present', category: 'schema', status: 'pass', message: '16/16 fields found' },
  { id: 'v2', label: 'Slug format valid', category: 'schema', status: 'pass', message: 'rsi_zones matches [a-z][a-z0-9_]*' },
  { id: 'v3', label: 'Pack type valid', category: 'schema', status: 'pass', message: 'tf_confluence is supported' },
  { id: 'v4', label: 'No builtin name collisions', category: 'schema', status: 'pass', message: 'No conflicts with 11 reserved prefixes' },
  { id: 'v5', label: 'Output descriptions match', category: 'schema', status: 'pass', message: '3 outputs, 3 descriptions' },
  { id: 'v6', label: 'Allowed imports only', category: 'safety', status: 'pass', message: 'pandas, numpy only' },
  { id: 'v7', label: 'No unsafe function calls', category: 'safety', status: 'pass', message: 'No exec/eval/open detected' },
  { id: 'v8', label: 'No unsafe module access', category: 'safety', status: 'pass', message: 'No os/sys/subprocess detected' },
  { id: 'v9', label: 'Indicator function exists', category: 'functions', status: 'pass', message: 'calculate_rsi_zones found in indicator.py' },
  { id: 'v10', label: 'Interpreter function exists', category: 'functions', status: 'pass', message: 'interpret_rsi_zones found' },
  { id: 'v11', label: 'Trigger function exists', category: 'functions', status: 'pass', message: 'detect_rsi_zones_triggers found' },
  { id: 'v12', label: 'Valid execution types', category: 'execution', status: 'pass', message: 'All triggers use valid exec types' },
  { id: 'v13', label: 'L-type triggers configured', category: 'execution', status: 'warn', message: 'cross_above_ib missing level_column — will default to indicator column' },
  { id: 'v14', label: 'Trigger key pattern valid', category: 'execution', status: 'pass', message: 'All keys match {prefix}_{base}' },
  { id: 'v15', label: 'Sample data produces signals', category: 'backtest', status: 'pending', message: 'Click "Run Backtest" to test' },
  { id: 'v16', label: 'Signal timing alignment', category: 'backtest', status: 'pending', message: 'Requires backtest run' },
];

/* ========================================================================= */
/* COMPONENT                                                                   */
/* ========================================================================= */

export default function PackBuilderV5() {
  // Wizard state
  const [step, setStep] = useState(1);

  // Step 1: Pack Type & Info
  const [packType, setPackType] = useState<PackType>('tf_confluence');
  const [packName, setPackName] = useState('');
  const [category, setCategory] = useState('');
  const [displayType, setDisplayType] = useState<string>('oscillator');
  const [description, setDescription] = useState('');
  const [version, setVersion] = useState('Default');

  // Step 2: Structure
  const [params, setParams] = useState<ParamDef[]>([
    { id: generateId(), name: 'period', label: 'RSI Period', type: 'int', defaultVal: '14', min: '1', max: '100' },
    { id: generateId(), name: 'overbought', label: 'Overbought', type: 'int', defaultVal: '70', min: '50', max: '95' },
    { id: generateId(), name: 'oversold', label: 'Oversold', type: 'int', defaultVal: '30', min: '5', max: '50' },
  ]);
  const [outputs, setOutputs] = useState<OutputDef[]>([
    { id: generateId(), code: 'OVERBOUGHT', description: 'RSI above overbought threshold' },
    { id: generateId(), code: 'NEUTRAL', description: 'RSI between thresholds' },
    { id: generateId(), code: 'OVERSOLD', description: 'RSI below oversold threshold' },
  ]);
  const [triggers, setTriggers] = useState<TriggerDef[]>([
    { id: generateId(), name: 'Cross Above Overbought', base: 'cross_above_ob', sentiment: 'bearish', execTypes: ['[C]', '[L]'], fidelity: '[PB]', fromState: 'NEUTRAL', toState: 'OVERBOUGHT' },
    { id: generateId(), name: 'Cross Below Oversold', base: 'cross_below_os', sentiment: 'bullish', execTypes: ['[C]', '[L]'], fidelity: '[PB]', fromState: 'NEUTRAL', toState: 'OVERSOLD' },
  ]);

  // Step 3: Prompt
  const [pineScript, setPineScript] = useState('');
  const [promptGenerated, setPromptGenerated] = useState(false);

  // Step 4: Paste & Validate
  const [llmResponse, setLlmResponse] = useState('');
  const [parsed, setParsed] = useState(false);
  const [validation, setValidation] = useState<ValidationItem[]>([]);

  // Derived
  const slug = useMemo(() => toSlug(packName), [packName]);
  const categories = packType === 'tf_confluence' ? TF_CATEGORIES : GEN_CATEGORIES;
  const canProceed1 = packName.trim().length > 0 && slug.length > 0 && category.length > 0;
  const canProceed2 = outputs.length >= 2 && params.length > 0;

  const STEPS = ['Pack Type & Info', 'Define Structure', 'Generate Prompt', 'Paste & Validate', 'Review & Install'];

  return (
    <div>
      <PageHeader title="Pack Builder" subtitle="Create custom confluence packs with AI assistance" />

      {/* ---- Wizard Stepper ---- */}
      <div className="flex items-center mb-6">
        {STEPS.map((s, i) => {
          const stepNum = i + 1;
          const isComplete = step > stepNum;
          const isCurrent = step === stepNum;
          return (
            <div key={i} className="flex items-center flex-1">
              <button
                className="flex items-center gap-2"
                style={{ cursor: isComplete ? 'pointer' : 'default', background: 'none', border: 'none', padding: 0 }}
                onClick={() => isComplete && setStep(stepNum)}
              >
                <span className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0" style={{
                  background: isComplete ? 'var(--green)' : isCurrent ? 'var(--accent)' : 'var(--bg-input)',
                  color: isComplete || isCurrent ? 'white' : 'var(--text-muted)',
                  border: isCurrent ? 'none' : isComplete ? 'none' : '1px solid var(--border)',
                }}>
                  {isComplete ? '✓' : stepNum}
                </span>
                <span className="text-xs font-medium hidden lg:inline" style={{ color: isCurrent ? 'var(--accent)' : isComplete ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                  {s}
                </span>
              </button>
              {i < STEPS.length - 1 && (
                <div className="flex-1 h-0.5 mx-2" style={{ background: isComplete ? 'var(--green)' : 'var(--border)' }} />
              )}
            </div>
          );
        })}
      </div>

      {/* ================================================================= */}
      {/* STEP 1: PACK TYPE & INFO                                          */}
      {/* ================================================================= */}
      {step === 1 && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <Card>
              {/* Pack type selector */}
              <p className="text-sm font-medium mb-3">Pack Type</p>
              <div className="grid grid-cols-2 gap-3 mb-5">
                {[
                  { type: 'tf_confluence' as PackType, name: 'TF Confluence', desc: 'Indicator-based with state classifiers and trade triggers', tags: 'Moving Averages, Momentum, Volume...' },
                  { type: 'general' as PackType, name: 'General', desc: 'Scalar time/session-based conditions', tags: 'Time, Session, Calendar...' },
                ].map((opt) => (
                  <button key={opt.type} className="text-left p-4 rounded-lg transition-colors"
                    style={{
                      background: packType === opt.type ? 'var(--accent-muted)' : 'var(--bg-input)',
                      border: packType === opt.type ? '2px solid var(--accent)' : '2px solid var(--border)',
                      cursor: 'pointer',
                    }}
                    onClick={() => { setPackType(opt.type); setCategory(''); }}>
                    <p className="text-sm font-semibold" style={{ color: packType === opt.type ? 'var(--accent)' : 'var(--text-primary)' }}>{opt.name}</p>
                    <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>{opt.desc}</p>
                    <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>{opt.tags}</p>
                  </button>
                ))}
              </div>

              {/* Form fields */}
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>Pack Name</label>
                  <input type="text" value={packName} onChange={(e) => setPackName(e.target.value)} placeholder="RSI Zones" style={inputStyle} />
                  {slug && <p className="text-[10px] font-mono mt-1" style={{ color: 'var(--text-muted)' }}>Slug: {slug}</p>}
                </div>
                <div>
                  <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>Version</label>
                  <input type="text" value={version} onChange={(e) => setVersion(e.target.value)} style={inputStyle} />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>Category</label>
                  <div className="flex flex-wrap gap-1">
                    {categories.map((c) => (
                      <button key={c} className="text-xs px-2.5 py-1 rounded-full" style={{
                        background: category === c ? 'var(--accent-muted)' : 'var(--bg-input)',
                        color: category === c ? 'var(--accent)' : 'var(--text-muted)',
                        border: category === c ? '1px solid var(--accent)' : '1px solid var(--border)',
                        cursor: 'pointer',
                      }} onClick={() => setCategory(c)}>{c}</button>
                    ))}
                  </div>
                </div>
                {packType === 'tf_confluence' && (
                  <div>
                    <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>Display Type</label>
                    <div className="flex gap-2">
                      {DISPLAY_TYPES.map((dt) => (
                        <button key={dt} className="text-xs px-3 py-1.5 rounded-lg capitalize" style={{
                          background: displayType === dt ? 'var(--accent-muted)' : 'var(--bg-input)',
                          color: displayType === dt ? 'var(--accent)' : 'var(--text-muted)',
                          border: displayType === dt ? '1px solid var(--accent)' : '1px solid var(--border)',
                          cursor: 'pointer',
                        }} onClick={() => setDisplayType(dt)}>{dt}</button>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <div>
                <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>Description</label>
                <textarea value={description} onChange={(e) => setDescription(e.target.value)} rows={3} placeholder="Describe what this pack does..."
                  style={{ ...inputStyle, resize: 'vertical' }} />
              </div>
            </Card>
          </div>

          {/* Info panel */}
          <Card>
            <h4 className="text-sm font-medium mb-3">What is a {packType === 'tf_confluence' ? 'TF Confluence' : 'General'} Pack?</h4>
            {packType === 'tf_confluence' ? (
              <div className="space-y-2 text-xs" style={{ color: 'var(--text-muted)' }}>
                <p>A TF Confluence pack computes technical indicators from price/volume data and classifies each bar into mutually exclusive states.</p>
                <p><strong>3 files generated:</strong></p>
                <ul className="list-disc pl-4 space-y-1">
                  <li><span className="font-mono">manifest.json</span> — metadata, parameters, outputs, triggers</li>
                  <li><span className="font-mono">indicator.py</span> — computes indicator values on DataFrame</li>
                  <li><span className="font-mono">interpreter.py</span> — classifies states + detects triggers</li>
                </ul>
                <p><strong>Execution types:</strong> <ExecBadge tag="[C]" /> <ExecBadge tag="[L]" /> <ExecBadge tag="[LC]" /> <ExecBadge tag="[CC]" /></p>
                <p><strong>Fidelity:</strong> <FidelityBadge tag="[PB]" /> Previous Bar &middot; <FidelityBadge tag="[CB]" /> Current Bar</p>
              </div>
            ) : (
              <div className="space-y-2 text-xs" style={{ color: 'var(--text-muted)' }}>
                <p>A General pack evaluates scalar conditions (time-of-day, trading session, calendar) that apply globally across all timeframes.</p>
                <p><strong>2 files generated:</strong></p>
                <ul className="list-disc pl-4 space-y-1">
                  <li><span className="font-mono">manifest.json</span> — metadata, parameters, outputs</li>
                  <li><span className="font-mono">evaluator.py</span> — evaluates condition per timestamp</li>
                </ul>
                <p><strong>Execution:</strong> <ExecBadge tag="[C]" /> Bar close only</p>
                <p><strong>Outputs:</strong> Binary (IN/OUT pattern)</p>
                <p><strong>No chart overlay</strong> — purely logical filters</p>
              </div>
            )}
          </Card>
        </div>
      )}

      {/* ================================================================= */}
      {/* STEP 2: DEFINE STRUCTURE                                          */}
      {/* ================================================================= */}
      {step === 2 && (
        <div className="space-y-6">
          {/* Parameters */}
          <Card>
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-medium">Parameters <span className="text-xs font-normal" style={{ color: 'var(--text-muted)' }}>({params.length})</span></h4>
              <button style={{ ...btnSecondary, fontSize: '0.75rem', padding: '4px 10px' }}
                onClick={() => setParams([...params, { id: generateId(), name: '', label: '', type: 'int', defaultVal: '', min: '', max: '' }])}>
                + Add
              </button>
            </div>
            <div className="space-y-2">
              {params.map((p, i) => (
                <div key={p.id} className="grid grid-cols-6 gap-2 items-end">
                  <div>
                    <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Name</label>
                    <input type="text" value={p.name} onChange={(e) => { const n = [...params]; n[i].name = e.target.value; setParams(n); }} style={{ ...inputStyle, fontSize: '0.75rem' }} />
                  </div>
                  <div>
                    <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Label</label>
                    <input type="text" value={p.label} onChange={(e) => { const n = [...params]; n[i].label = e.target.value; setParams(n); }} style={{ ...inputStyle, fontSize: '0.75rem' }} />
                  </div>
                  <div>
                    <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Type</label>
                    <select value={p.type} onChange={(e) => { const n = [...params]; n[i].type = e.target.value as ParamDef['type']; setParams(n); }} style={{ ...inputStyle, fontSize: '0.75rem' }}>
                      <option value="int">int</option>
                      <option value="float">float</option>
                      {packType === 'general' && <option value="bool">bool</option>}
                      {packType === 'general' && <option value="select">select</option>}
                    </select>
                  </div>
                  <div>
                    <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Default</label>
                    <input type="text" value={p.defaultVal} onChange={(e) => { const n = [...params]; n[i].defaultVal = e.target.value; setParams(n); }} style={{ ...inputStyle, fontSize: '0.75rem' }} />
                  </div>
                  <div>
                    <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Min / Max</label>
                    <div className="flex gap-1">
                      <input type="text" value={p.min} onChange={(e) => { const n = [...params]; n[i].min = e.target.value; setParams(n); }} placeholder="min" style={{ ...inputStyle, fontSize: '0.75rem', width: '50%' }} />
                      <input type="text" value={p.max} onChange={(e) => { const n = [...params]; n[i].max = e.target.value; setParams(n); }} placeholder="max" style={{ ...inputStyle, fontSize: '0.75rem', width: '50%' }} />
                    </div>
                  </div>
                  <button className="text-xs px-2 py-1 rounded" style={{ color: 'var(--red)', cursor: 'pointer', background: 'transparent', border: 'none' }}
                    onClick={() => setParams(params.filter((_, j) => j !== i))}>Remove</button>
                </div>
              ))}
            </div>
          </Card>

          {/* Outputs */}
          <Card>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <h4 className="text-sm font-medium">Outputs <span className="text-xs font-normal" style={{ color: 'var(--text-muted)' }}>({outputs.length})</span></h4>
                {packType === 'tf_confluence' && <FidelityBadge tag="[PB]" />}
              </div>
              <button style={{ ...btnSecondary, fontSize: '0.75rem', padding: '4px 10px' }}
                onClick={() => setOutputs([...outputs, { id: generateId(), code: '', description: '' }])}>
                + Add
              </button>
            </div>
            <div className="space-y-2">
              {outputs.map((o, i) => (
                <div key={o.id} className="flex items-center gap-3">
                  <input type="text" value={o.code} onChange={(e) => { const n = [...outputs]; n[i].code = e.target.value.toUpperCase(); setOutputs(n); }}
                    placeholder="STATE_NAME" className="font-mono" style={{ ...inputStyle, fontSize: '0.75rem', width: 160 }} />
                  <input type="text" value={o.description} onChange={(e) => { const n = [...outputs]; n[i].description = e.target.value; setOutputs(n); }}
                    placeholder="Description..." style={{ ...inputStyle, fontSize: '0.75rem', flex: 1 }} />
                  <button className="text-xs" style={{ color: 'var(--red)', cursor: 'pointer', background: 'transparent', border: 'none' }}
                    onClick={() => setOutputs(outputs.filter((_, j) => j !== i))}>Remove</button>
                </div>
              ))}
            </div>
            {packType === 'general' && outputs.length !== 2 && (
              <p className="text-xs mt-2" style={{ color: 'var(--orange)' }}>General packs require exactly 2 outputs (IN/OUT pattern)</p>
            )}
          </Card>

          {/* Triggers */}
          <Card>
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-medium">Triggers <span className="text-xs font-normal" style={{ color: 'var(--text-muted)' }}>({triggers.length})</span></h4>
              <button style={{ ...btnSecondary, fontSize: '0.75rem', padding: '4px 10px' }}
                onClick={() => setTriggers([...triggers, { id: generateId(), name: '', base: '', sentiment: 'neutral', execTypes: ['[C]'], fidelity: '[PB]', fromState: '', toState: '' }])}>
                + Add
              </button>
            </div>
            <div className="space-y-3">
              {triggers.map((t, i) => (
                <div key={t.id} className="rounded-lg p-3" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                  <div className="grid grid-cols-3 gap-3 mb-2">
                    <div>
                      <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Trigger Name</label>
                      <input type="text" value={t.name} onChange={(e) => { const n = [...triggers]; n[i].name = e.target.value; n[i].base = toSlug(e.target.value); setTriggers(n); }}
                        placeholder="Cross Above Level" style={{ ...inputStyle, fontSize: '0.75rem' }} />
                      {t.base && <p className="text-[10px] font-mono mt-0.5" style={{ color: 'var(--text-muted)' }}>Base: {t.base}</p>}
                    </div>
                    <div>
                      <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Sentiment</label>
                      <div className="flex gap-1">
                        {SENTIMENTS.map((s) => (
                          <button key={s} className="text-[10px] px-2 py-1 rounded capitalize" style={{
                            background: t.sentiment === s ? (s === 'bullish' ? 'var(--green-muted)' : s === 'bearish' ? 'var(--red-muted)' : 'var(--bg-card)') : 'transparent',
                            color: t.sentiment === s ? (s === 'bullish' ? 'var(--green)' : s === 'bearish' ? 'var(--red)' : 'var(--text-muted)') : 'var(--text-muted)',
                            border: t.sentiment === s ? 'none' : '1px solid var(--border)', cursor: 'pointer',
                          }} onClick={() => { const n = [...triggers]; n[i].sentiment = s; setTriggers(n); }}>{s}</button>
                        ))}
                      </div>
                    </div>
                    <div>
                      <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>State Transition</label>
                      <div className="flex items-center gap-1">
                        <select value={t.fromState} onChange={(e) => { const n = [...triggers]; n[i].fromState = e.target.value; setTriggers(n); }}
                          style={{ ...inputStyle, fontSize: '0.7rem', width: '45%' }}>
                          <option value="">From...</option>
                          {outputs.map((o) => <option key={o.id} value={o.code}>{o.code}</option>)}
                        </select>
                        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>→</span>
                        <select value={t.toState} onChange={(e) => { const n = [...triggers]; n[i].toState = e.target.value; setTriggers(n); }}
                          style={{ ...inputStyle, fontSize: '0.7rem', width: '45%' }}>
                          <option value="">To...</option>
                          {outputs.map((o) => <option key={o.id} value={o.code}>{o.code}</option>)}
                        </select>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    {/* Exec types */}
                    <div className="flex items-center gap-1">
                      <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Exec:</span>
                      {EXEC_TYPES.map((et) => {
                        const disabled = packType === 'general' && et !== '[C]';
                        return (
                          <label key={et} className="flex items-center gap-0.5" style={{ opacity: disabled ? 0.3 : 1 }}>
                            <input type="checkbox" disabled={disabled}
                              checked={t.execTypes.includes(et)}
                              onChange={(e) => {
                                const n = [...triggers];
                                if (e.target.checked) n[i].execTypes = [...n[i].execTypes, et];
                                else n[i].execTypes = n[i].execTypes.filter((x) => x !== et);
                                setTriggers(n);
                              }}
                              className="w-3 h-3" style={{ accentColor: EXEC_BADGE_COLOR }} />
                            <ExecBadge tag={et} />
                          </label>
                        );
                      })}
                    </div>
                    {/* Fidelity */}
                    {packType === 'tf_confluence' && (
                      <div className="flex items-center gap-1">
                        <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Fidelity:</span>
                        {['[PB]', '[CB]'].map((f) => (
                          <button key={f} className="text-[10px] font-mono px-1.5 py-0.5 rounded-full" style={{
                            color: t.fidelity === f ? FIDELITY_BADGE_COLOR : 'var(--text-muted)',
                            background: t.fidelity === f ? FIDELITY_BADGE_COLOR + '20' : 'transparent',
                            border: t.fidelity === f ? 'none' : '1px solid var(--border)', cursor: 'pointer',
                          }} onClick={() => { const n = [...triggers]; n[i].fidelity = f; setTriggers(n); }}>{f}</button>
                        ))}
                      </div>
                    )}
                    <span className="flex-1" />
                    <button className="text-xs" style={{ color: 'var(--red)', cursor: 'pointer', background: 'transparent', border: 'none' }}
                      onClick={() => setTriggers(triggers.filter((_, j) => j !== i))}>Remove</button>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}

      {/* ================================================================= */}
      {/* STEP 3: GENERATE & COPY PROMPT                                    */}
      {/* ================================================================= */}
      {step === 3 && (
        <div>
          {/* Summary banner */}
          <Card className="mb-4">
            <div className="flex items-center gap-6 text-xs" style={{ color: 'var(--text-muted)' }}>
              <span><strong style={{ color: 'var(--text-primary)' }}>{packName || 'Unnamed Pack'}</strong></span>
              <span>{packType === 'tf_confluence' ? 'TF Confluence' : 'General'}</span>
              <span>{params.length} params</span>
              <span>{outputs.length} outputs</span>
              <span>{triggers.length} triggers</span>
            </div>
          </Card>

          {/* Pine Script (optional) */}
          <Card className="mb-4">
            <details>
              <summary className="text-xs font-medium cursor-pointer" style={{ color: 'var(--text-muted)' }}>
                Optional: Paste TradingView Pine Script for translation context
              </summary>
              <textarea value={pineScript} onChange={(e) => setPineScript(e.target.value)} rows={6} placeholder="// Paste your Pine Script v5 code here..."
                className="w-full mt-2 font-mono" style={{ ...inputStyle, fontSize: '0.75rem' }} />
            </details>
          </Card>

          {/* Generated prompt */}
          <Card>
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-medium">Generated Prompt</h4>
              <div className="flex gap-2">
                <button style={btnPrimary} onClick={() => setPromptGenerated(true)}>
                  {promptGenerated ? 'Copy to Clipboard' : 'Generate Prompt'}
                </button>
                {promptGenerated && <button style={btnSecondary} onClick={() => setPromptGenerated(true)}>Regenerate</button>}
              </div>
            </div>
            {promptGenerated ? (
              <div className="rounded-lg p-4 font-mono text-xs" style={{ background: '#0d1117', color: '#c9d1d9', maxHeight: 400, overflowY: 'auto', lineHeight: 1.6 }}>
                <p style={{ color: '#58a6ff' }}>// Pack Builder Prompt — {packName}</p>
                <p style={{ color: '#8b949e' }}>// Generated {new Date().toLocaleString()}</p>
                <br />
                <p>You are an expert Python developer specializing in quantitative trading indicators.</p>
                <p>Create a confluence pack called &quot;{packName}&quot; ({packType}) with the following specification:</p>
                <br />
                <p style={{ color: '#d2a8ff' }}>## Parameters</p>
                {params.map((p) => <p key={p.id}>- {p.name} ({p.type}): default={p.defaultVal}, range=[{p.min}, {p.max}] — {p.label}</p>)}
                <br />
                <p style={{ color: '#d2a8ff' }}>## Outputs (mutually exclusive states)</p>
                {outputs.map((o) => <p key={o.id}>- {o.code}: {o.description}</p>)}
                <br />
                <p style={{ color: '#d2a8ff' }}>## Triggers</p>
                {triggers.map((t) => <p key={t.id}>- {t.base}: {t.name} ({t.sentiment}, {t.execTypes.join('/')}, {t.fromState} → {t.toState})</p>)}
                <br />
                <p style={{ color: '#8b949e' }}>// ... [full architecture context from pack_builder_context.md would be appended here] ...</p>
              </div>
            ) : (
              <div className="rounded-lg p-8 text-center" style={{ background: 'var(--bg-input)' }}>
                <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Click &quot;Generate Prompt&quot; to create the AI prompt based on your pack specification.</p>
              </div>
            )}
            {promptGenerated && (
              <p className="text-xs mt-3" style={{ color: 'var(--text-muted)' }}>
                Copy this prompt and paste it into Claude, ChatGPT, or any AI assistant. Copy the full response including all code blocks back to Step 4.
              </p>
            )}
          </Card>
        </div>
      )}

      {/* ================================================================= */}
      {/* STEP 4: PASTE & VALIDATE                                          */}
      {/* ================================================================= */}
      {step === 4 && (
        <div className="grid grid-cols-12 gap-4">
          {/* Left: paste + code preview */}
          <div className="col-span-7">
            <Card className="mb-4">
              <h4 className="text-sm font-medium mb-2">Paste LLM Response</h4>
              <textarea value={llmResponse} onChange={(e) => setLlmResponse(e.target.value)} rows={8} placeholder="Paste the full AI response including all code blocks..."
                className="w-full font-mono" style={{ ...inputStyle, fontSize: '0.75rem' }} />
              <button className="mt-2" style={{ ...btnPrimary, opacity: llmResponse.trim() ? 1 : 0.4 }}
                disabled={!llmResponse.trim()}
                onClick={() => { setParsed(true); setValidation(mockValidation); }}>
                Parse Response
              </button>
            </Card>

            {parsed && (
              <Card>
                <h4 className="text-sm font-medium mb-3">Parsed Code Preview</h4>
                <TabBar tabs={packType === 'tf_confluence' ? ['manifest.json', 'indicator.py', 'interpreter.py'] : ['manifest.json', 'evaluator.py']}>
                  {(tab) => (
                    <div className="rounded-lg p-4 font-mono text-xs" style={{ background: '#0d1117', color: '#c9d1d9', maxHeight: 300, overflowY: 'auto', lineHeight: 1.6 }}>
                      {tab === 'manifest.json' && (
                        <pre>{`{\n  "slug": "${slug}",\n  "name": "${packName}",\n  "pack_type": "${packType}",\n  "category": "${category}",\n  "outputs": ${JSON.stringify(outputs.map(o => o.code))},\n  "triggers": [...],\n  "parameters_schema": {...}\n}`}</pre>
                      )}
                      {tab === 'indicator.py' && (
                        <pre>{`import pandas as pd\nimport numpy as np\n\ndef calculate_${slug}(df: pd.DataFrame, ${params.map(p => `${p.name}=${p.defaultVal}`).join(', ')}) -> pd.DataFrame:\n    """Calculate ${packName} indicator."""\n    # ... indicator logic ...\n    return df`}</pre>
                      )}
                      {tab === 'interpreter.py' && (
                        <pre>{`def interpret_${slug}(df: pd.DataFrame, **params) -> pd.Series:\n    """Classify ${packName} states."""\n    # ... classification logic ...\n    pass\n\ndef detect_${slug}_triggers(df: pd.DataFrame, **params) -> dict:\n    """Detect ${packName} trigger events."""\n    return {\n${triggers.map(t => `        "${t.base}": ...`).join(',\n')}\n    }`}</pre>
                      )}
                      {tab === 'evaluator.py' && (
                        <pre>{`def evaluate_${slug}(timestamp, ${params.map(p => `${p.name}=${p.defaultVal}`).join(', ')}):\n    """Evaluate ${packName} condition."""\n    # ... scalar evaluation ...\n    return "IN_WINDOW"  # or "OUT_OF_WINDOW"`}</pre>
                      )}
                    </div>
                  )}
                </TabBar>
              </Card>
            )}
          </div>

          {/* Right: validation checklist */}
          <div className="col-span-5">
            <Card>
              <div className="flex items-center justify-between mb-3">
                <h4 className="text-sm font-medium">Validation</h4>
                {validation.length > 0 && (
                  <span className="text-xs font-bold" style={{ color: validation.every(v => v.status === 'pass' || v.status === 'pending') ? 'var(--green)' : 'var(--red)' }}>
                    {validation.filter(v => v.status === 'pass').length}/{validation.length} passing
                  </span>
                )}
              </div>

              {validation.length === 0 ? (
                <p className="text-xs py-8 text-center" style={{ color: 'var(--text-muted)' }}>Parse a response to run validation</p>
              ) : (
                <div className="space-y-3">
                  {['schema', 'safety', 'functions', 'execution', 'backtest'].map((cat) => {
                    const items = validation.filter(v => v.category === cat);
                    if (items.length === 0) return null;
                    const allPass = items.every(v => v.status === 'pass' || v.status === 'pending');
                    return (
                      <div key={cat}>
                        <p className="text-xs font-medium mb-1 capitalize" style={{ color: allPass ? 'var(--green)' : 'var(--text-secondary)' }}>{cat}</p>
                        <div className="space-y-1">
                          {items.map((v) => (
                            <div key={v.id} className="flex items-start gap-2 text-xs">
                              <StatusIcon status={v.status} />
                              <div>
                                <span style={{ color: 'var(--text-secondary)' }}>{v.label}</span>
                                <p className="text-[10px]" style={{ color: v.status === 'fail' ? 'var(--red)' : v.status === 'warn' ? 'var(--orange)' : 'var(--text-muted)' }}>{v.message}</p>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                  <button className="w-full mt-2" style={{ ...btnSecondary, fontSize: '0.75rem' }}>Run Backtest Parity</button>
                </div>
              )}
            </Card>
          </div>
        </div>
      )}

      {/* ================================================================= */}
      {/* STEP 5: REVIEW & INSTALL                                          */}
      {/* ================================================================= */}
      {step === 5 && (
        <div>
          <TabBar tabs={['Overview', 'Chart Preview', 'Code', 'Test Results']}>
            {(tab) => (
              <div>
                {tab === 'Overview' && (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    <Card>
                      <h4 className="text-sm font-medium mb-3">Pack Summary</h4>
                      <div className="space-y-2 text-xs">
                        {[
                          { label: 'Name', value: packName || 'RSI Zones' },
                          { label: 'Slug', value: slug || 'rsi_zones' },
                          { label: 'Type', value: packType === 'tf_confluence' ? 'TF Confluence' : 'General' },
                          { label: 'Category', value: category || 'Momentum' },
                          { label: 'Display', value: displayType },
                          { label: 'Version', value: version },
                        ].map((r) => (
                          <div key={r.label} className="flex justify-between">
                            <span style={{ color: 'var(--text-muted)' }}>{r.label}</span>
                            <span className="font-medium">{r.value}</span>
                          </div>
                        ))}
                      </div>
                    </Card>
                    <Card>
                      <h4 className="text-sm font-medium mb-3">Triggers</h4>
                      <div className="space-y-2">
                        {triggers.map((t) => (
                          <div key={t.id} className="flex items-center gap-2 flex-wrap">
                            <SentimentBadge sentiment={t.sentiment} />
                            <span className="text-xs">{t.name}</span>
                            {t.execTypes.map((et) => <ExecBadge key={et} tag={et} />)}
                            {packType === 'tf_confluence' && <FidelityBadge tag={t.fidelity} />}
                            {t.fromState && t.toState && (
                              <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>{t.fromState} → {t.toState}</span>
                            )}
                          </div>
                        ))}
                      </div>
                    </Card>
                  </div>
                )}
                {tab === 'Chart Preview' && (
                  <Card>
                    <ChartPlaceholder label={`${packType === 'tf_confluence' ? (displayType === 'oscillator' ? 'Oscillator chart with overbought/oversold zones and signal markers' : 'Price chart with indicator overlay lines and signal markers') : 'Condition timeline: horizontal bars showing IN/OUT periods over 90 days'}`} height={350} />
                  </Card>
                )}
                {tab === 'Code' && (
                  <Card>
                    <div className="space-y-3">
                      {(packType === 'tf_confluence' ? ['manifest.json', 'indicator.py', 'interpreter.py'] : ['manifest.json', 'evaluator.py']).map((file) => (
                        <details key={file}>
                          <summary className="text-xs font-mono font-medium cursor-pointer py-2" style={{ color: 'var(--accent)' }}>{file}</summary>
                          <div className="rounded-lg p-3 font-mono text-xs mt-1" style={{ background: '#0d1117', color: '#c9d1d9', maxHeight: 250, overflowY: 'auto' }}>
                            <pre>// {file} content would appear here after parsing</pre>
                          </div>
                        </details>
                      ))}
                    </div>
                  </Card>
                )}
                {tab === 'Test Results' && (
                  <Card>
                    <h4 className="text-sm font-medium mb-3">Signal Test Results</h4>
                    <div className="grid grid-cols-3 gap-4 mb-4">
                      {[
                        { label: 'Total Signals', value: '47' },
                        { label: 'Avg Bars Between', value: '12.3' },
                        { label: 'State Coverage', value: '98.2%' },
                      ].map((m) => (
                        <div key={m.label}>
                          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{m.label}</p>
                          <p className="text-lg font-bold">{m.value}</p>
                        </div>
                      ))}
                    </div>
                    <ChartPlaceholder label="Signal timeline: dots on a timeline showing when each trigger fired, colored by sentiment" height={150} />
                  </Card>
                )}
              </div>
            )}
          </TabBar>

          {/* Install actions */}
          <div className="flex items-center gap-3 mt-6">
            <button style={btnPrimary}>Install Pack</button>
            <button style={btnSecondary}>Save Draft</button>
            <button style={btnSecondary}>Export JSON</button>
          </div>
        </div>
      )}

      {/* ---- Navigation buttons ---- */}
      <div className="flex justify-between mt-6">
        {step > 1 ? (
          <button style={btnSecondary} onClick={() => setStep(step - 1)}>← Back</button>
        ) : <div />}
        {step < 5 && (
          <button style={btnPrimary}
            disabled={step === 1 && !canProceed1 || step === 2 && !canProceed2}
            onClick={() => setStep(step + 1)}>
            Next →
          </button>
        )}
      </div>
    </div>
  );
}
