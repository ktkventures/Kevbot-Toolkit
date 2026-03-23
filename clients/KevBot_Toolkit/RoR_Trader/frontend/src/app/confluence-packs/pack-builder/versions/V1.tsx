'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import ChartPlaceholder from '@/components/ChartPlaceholder';

interface TriggerDef {
  id: string;
  name: string;
  fromState: string;
  toState: string;
  execType: string;
}

const mockTriggers: TriggerDef[] = [
  { id: 't1', name: 'RSI Cross Above OB', fromState: 'NEUTRAL', toState: 'OVERBOUGHT', execType: '[C]' },
  { id: 't2', name: 'RSI Cross Below OS', fromState: 'NEUTRAL', toState: 'OVERSOLD', execType: '[C]' },
];

const mockOutputs = ['OVERBOUGHT', 'NEUTRAL', 'OVERSOLD'];

const execTypeOptions = ['[C]', '[L0]', '[L1]', '[HM]', '[HL]'];
const execTypeColors: Record<string, { color: string; bg: string }> = {
  '[C]': { color: 'var(--green)', bg: 'var(--green-muted)' },
  '[L0]': { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  '[L1]': { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  '[HM]': { color: 'var(--orange)', bg: 'var(--orange-muted)' },
  '[HL]': { color: 'var(--accent)', bg: 'var(--accent-muted)' },
};

export default function PackBuilderV1() {
  const [packName, setPackName] = useState('My Custom Pack');
  const [category, setCategory] = useState('Momentum');
  const [description, setDescription] = useState('');
  const [triggers, setTriggers] = useState<TriggerDef[]>(mockTriggers);

  function addTrigger() {
    setTriggers((prev) => [
      ...prev,
      {
        id: `t${Date.now()}`,
        name: '',
        fromState: '',
        toState: '',
        execType: '[C]',
      },
    ]);
  }

  function removeTrigger(id: string) {
    setTriggers((prev) => prev.filter((t) => t.id !== id));
  }

  function updateTrigger(id: string, field: keyof TriggerDef, value: string) {
    setTriggers((prev) =>
      prev.map((t) => (t.id === id ? { ...t, [field]: value } : t))
    );
  }

  return (
    <div>
      <PageHeader
        title="Pack Builder"
        subtitle="Build custom indicator packs with your own logic"
        actions={
          <div className="flex gap-2">
            <button
              className="px-4 py-2 rounded-lg text-sm"
              style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
            >
              Save Draft
            </button>
            <button
              className="px-4 py-2 rounded-lg text-sm font-medium"
              style={{ background: 'var(--accent)', color: 'white' }}
            >
              Publish Pack
            </button>
          </div>
        }
      />

      <div className="grid grid-cols-2 gap-6">
        {/* Left column — Config */}
        <div className="space-y-6">
          {/* Pack Info */}
          <Card>
            <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Pack Info</h3>
            <div className="space-y-4">
              <div>
                <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>Name</label>
                <input
                  className="w-full px-3 py-2 rounded-lg text-sm"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                  value={packName}
                  onChange={(e) => setPackName(e.target.value)}
                />
              </div>
              <div>
                <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>Category</label>
                <select
                  className="w-full px-3 py-2 rounded-lg text-sm"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                  value={category}
                  onChange={(e) => setCategory(e.target.value)}
                >
                  {['Moving Averages', 'Momentum', 'Volume', 'Volatility', 'Trend', 'Custom'].map((cat) => (
                    <option key={cat} value={cat}>{cat}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>Description</label>
                <textarea
                  className="w-full px-3 py-2 rounded-lg text-sm"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', resize: 'vertical' }}
                  rows={3}
                  placeholder="Describe what this pack does..."
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                />
              </div>
            </div>
          </Card>

          {/* Indicator Logic */}
          <Card>
            <h3 className="text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Indicator Logic</h3>
            <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>Define your indicator computation</p>
            <div
              className="rounded-lg p-4"
              style={{
                background: '#0d1117',
                border: '1px solid var(--border)',
                fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',
                minHeight: 180,
              }}
            >
              <p className="text-xs" style={{ color: '#8b949e' }}>
                <span style={{ color: '#ff7b72' }}>def</span>{' '}
                <span style={{ color: '#d2a8ff' }}>compute</span>
                <span style={{ color: '#c9d1d9' }}>(self, bar):</span>
              </p>
              <p className="text-xs mt-1" style={{ color: '#8b949e' }}>
                &nbsp;&nbsp;<span style={{ color: '#8b949e' }}># Calculate RSI from close prices</span>
              </p>
              <p className="text-xs" style={{ color: '#c9d1d9' }}>
                &nbsp;&nbsp;delta = bar.close - self.prev_close
              </p>
              <p className="text-xs" style={{ color: '#c9d1d9' }}>
                &nbsp;&nbsp;gain = <span style={{ color: '#79c0ff' }}>max</span>(delta, <span style={{ color: '#79c0ff' }}>0</span>)
              </p>
              <p className="text-xs" style={{ color: '#c9d1d9' }}>
                &nbsp;&nbsp;loss = <span style={{ color: '#79c0ff' }}>abs</span>(<span style={{ color: '#79c0ff' }}>min</span>(delta, <span style={{ color: '#79c0ff' }}>0</span>))
              </p>
              <p className="text-xs mt-2" style={{ color: '#c9d1d9' }}>
                &nbsp;&nbsp;<span style={{ color: '#ff7b72' }}>return</span> self._ema_smooth(gain, loss)
              </p>
            </div>
          </Card>

          {/* Interpreter Logic */}
          <Card>
            <h3 className="text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Interpreter Logic</h3>
            <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>Define state classification rules</p>
            <div
              className="rounded-lg p-4"
              style={{
                background: '#0d1117',
                border: '1px solid var(--border)',
                fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',
                minHeight: 140,
              }}
            >
              <p className="text-xs" style={{ color: '#8b949e' }}>
                <span style={{ color: '#ff7b72' }}>def</span>{' '}
                <span style={{ color: '#d2a8ff' }}>classify</span>
                <span style={{ color: '#c9d1d9' }}>(self, rsi_value):</span>
              </p>
              <p className="text-xs mt-1" style={{ color: '#c9d1d9' }}>
                &nbsp;&nbsp;<span style={{ color: '#ff7b72' }}>if</span> rsi_value &gt;= self.params[<span style={{ color: '#a5d6ff' }}>&apos;OB&apos;</span>]:
              </p>
              <p className="text-xs" style={{ color: '#c9d1d9' }}>
                &nbsp;&nbsp;&nbsp;&nbsp;<span style={{ color: '#ff7b72' }}>return</span> <span style={{ color: '#a5d6ff' }}>&apos;OVERBOUGHT&apos;</span>
              </p>
              <p className="text-xs" style={{ color: '#c9d1d9' }}>
                &nbsp;&nbsp;<span style={{ color: '#ff7b72' }}>elif</span> rsi_value &lt;= self.params[<span style={{ color: '#a5d6ff' }}>&apos;OS&apos;</span>]:
              </p>
              <p className="text-xs" style={{ color: '#c9d1d9' }}>
                &nbsp;&nbsp;&nbsp;&nbsp;<span style={{ color: '#ff7b72' }}>return</span> <span style={{ color: '#a5d6ff' }}>&apos;OVERSOLD&apos;</span>
              </p>
              <p className="text-xs" style={{ color: '#c9d1d9' }}>
                &nbsp;&nbsp;<span style={{ color: '#ff7b72' }}>return</span> <span style={{ color: '#a5d6ff' }}>&apos;NEUTRAL&apos;</span>
              </p>
            </div>
          </Card>

          {/* Triggers */}
          <Card>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Triggers</h3>
              <button
                onClick={addTrigger}
                className="px-3 py-1.5 rounded text-xs font-medium"
                style={{ background: 'var(--accent)', color: 'white' }}
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
                  <div className="flex items-center gap-2 mb-2">
                    <input
                      className="px-2 py-1 rounded text-sm flex-1"
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
                  <div className="flex items-center gap-2">
                    <div className="flex-1">
                      <label className="text-xs mb-0.5 block" style={{ color: 'var(--text-muted)' }}>From State</label>
                      <input
                        className="w-full px-2 py-1 rounded text-xs"
                        style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                        placeholder="e.g. NEUTRAL"
                        value={trigger.fromState}
                        onChange={(e) => updateTrigger(trigger.id, 'fromState', e.target.value)}
                      />
                    </div>
                    <span className="text-xs mt-4" style={{ color: 'var(--text-muted)' }}>&rarr;</span>
                    <div className="flex-1">
                      <label className="text-xs mb-0.5 block" style={{ color: 'var(--text-muted)' }}>To State</label>
                      <input
                        className="w-full px-2 py-1 rounded text-xs"
                        style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                        placeholder="e.g. OVERBOUGHT"
                        value={trigger.toState}
                        onChange={(e) => updateTrigger(trigger.id, 'toState', e.target.value)}
                      />
                    </div>
                    <div className="w-20">
                      <label className="text-xs mb-0.5 block" style={{ color: 'var(--text-muted)' }}>Type</label>
                      <select
                        className="w-full px-2 py-1 rounded text-xs"
                        style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                        value={trigger.execType}
                        onChange={(e) => updateTrigger(trigger.id, 'execType', e.target.value)}
                      >
                        {execTypeOptions.map((et) => (
                          <option key={et} value={et}>{et}</option>
                        ))}
                      </select>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* Right column — Preview */}
        <div className="space-y-6">
          {/* Live Preview */}
          <Card>
            <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Live Preview</h3>
            <div className="flex gap-2 mb-3">
              <select
                className="px-3 py-1.5 rounded-lg text-xs"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
              >
                <option>NVDA</option>
                <option>SPY</option>
                <option>AAPL</option>
              </select>
              <select
                className="px-3 py-1.5 rounded-lg text-xs"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
              >
                <option>5Min</option>
                <option>1Min</option>
                <option>15Min</option>
              </select>
            </div>
            <ChartPlaceholder label="Price chart with indicator overlay" height={300} />
          </Card>

          {/* Outputs */}
          <Card>
            <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Detected Outputs</h3>
            <div className="space-y-2">
              {mockOutputs.map((output) => (
                <div
                  key={output}
                  className="flex items-center justify-between px-3 py-2 rounded-lg"
                  style={{ background: 'var(--bg-input)' }}
                >
                  <span className="text-sm font-mono" style={{ color: 'var(--text-primary)' }}>{output}</span>
                  <span className="text-xs" style={{ color: 'var(--green)' }}>detected</span>
                </div>
              ))}
            </div>
          </Card>

          {/* Test Results */}
          <Card>
            <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Test Results</h3>
            <div className="grid grid-cols-2 gap-3 mb-4">
              <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)' }}>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Trigger Fires</p>
                <p className="text-lg font-semibold">47</p>
              </div>
              <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)' }}>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Bars Tested</p>
                <p className="text-lg font-semibold">2,400</p>
              </div>
              <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)' }}>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Fire Rate</p>
                <p className="text-lg font-semibold">2.0%</p>
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
                    <span
                      className="text-xs font-mono px-1.5 py-0.5 rounded"
                      style={{
                        color: execTypeColors[trigger.execType]?.color || 'var(--text-muted)',
                        background: execTypeColors[trigger.execType]?.bg || 'var(--bg-input)',
                      }}
                    >
                      {trigger.execType}
                    </span>
                  </div>
                  <div className="flex items-center gap-4">
                    <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                      {Math.floor(Math.random() * 30 + 10)} fires
                    </span>
                    <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                      {(Math.random() * 3 + 0.5).toFixed(1)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
