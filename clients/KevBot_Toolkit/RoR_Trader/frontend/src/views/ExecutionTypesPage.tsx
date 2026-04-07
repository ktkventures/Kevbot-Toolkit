'use client';

import { useState } from 'react';
import dynamic from 'next/dynamic';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';

const SyncedChartPane = dynamic(() => import('@/charts/SyncedChartPane'), { ssr: false });
import { useExecutionTypes, useToggleExecutionType, useUpdateExecTypeParams, useSimulateExecType, useCreateVariation, useDeleteVariation, type ExecTypeModule, type ExecTypeVariation, type SimulationResult } from '@/hooks/queries/useExecutionTypes';

/* ========================================================================= */
/* STYLES                                                                      */
/* ========================================================================= */

const EXEC_BADGE_COLOR = '#2196F3';
const CONTEXT_COLORS: Record<string, { color: string; bg: string }> = {
  entry: { color: 'var(--green)', bg: 'var(--green-muted)' },
  exit_signal: { color: 'var(--red)', bg: 'var(--red-muted)' },
  stop: { color: 'var(--orange)', bg: 'var(--orange-muted)' },
  target: { color: 'var(--accent)', bg: 'var(--accent-muted)' },
};

const inputStyle: React.CSSProperties = {
  background: 'var(--bg-input)', border: '1px solid var(--border)',
  color: 'var(--text-primary)', padding: '6px 10px', borderRadius: '8px',
  fontSize: '0.8rem', width: '100%',
};
const btnPrimary: React.CSSProperties = {
  background: 'var(--accent)', color: 'white', border: 'none',
  padding: '8px 16px', borderRadius: '8px', fontSize: '0.875rem',
  cursor: 'pointer', fontWeight: 600,
};
const btnSecondary: React.CSSProperties = {
  background: 'var(--bg-card)', border: '1px solid var(--border)',
  color: 'var(--text-secondary)', padding: '6px 14px', borderRadius: '8px',
  fontSize: '0.875rem', cursor: 'pointer',
};

/* ========================================================================= */
/* SUB-COMPONENTS                                                              */
/* ========================================================================= */

function Toggle({ enabled, onToggle }: { enabled: boolean; onToggle: () => void }) {
  return (
    <button onClick={(e) => { e.stopPropagation(); onToggle(); }}
      className="w-10 h-6 rounded-full relative flex-shrink-0 transition-colors"
      style={{ background: enabled ? 'var(--accent)' : 'var(--bg-input)', border: enabled ? 'none' : '1px solid var(--border)' }}>
      <div className="w-4 h-4 rounded-full absolute top-1 transition-all"
        style={{ background: enabled ? 'white' : 'var(--text-muted)', left: enabled ? '22px' : '4px' }} />
    </button>
  );
}

function ExecCard({ mod, onToggle, onDetails }: { mod: ExecTypeModule; onToggle: () => void; onDetails: () => void }) {
  return (
    <Card>
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2 cursor-pointer" onClick={onDetails}>
          <span className="text-xs font-mono font-bold px-2 py-1 rounded-full"
            style={{ color: EXEC_BADGE_COLOR, background: EXEC_BADGE_COLOR + '20' }}>
            [{mod.display_code}]
          </span>
          <h3 className="text-sm font-semibold">{mod.name}</h3>
        </div>
        <Toggle enabled={mod.enabled} onToggle={onToggle} />
      </div>
      <p className="text-xs mb-3 cursor-pointer" style={{ color: 'var(--text-muted)' }} onClick={onDetails}>{mod.description}</p>
      <div className="flex items-center justify-between">
        <div className="flex gap-1 flex-wrap">
          {mod.contexts.map((ctx) => {
            const colors = CONTEXT_COLORS[ctx] || { color: 'var(--text-muted)', bg: 'var(--bg-input)' };
            return (
              <span key={ctx} className="text-[9px] px-1.5 py-0.5 rounded-full capitalize"
                style={{ color: colors.color, background: colors.bg }}>
                {ctx.replace(/_/g, ' ')}
              </span>
            );
          })}
        </div>
        <button className="text-xs px-3 py-1 rounded-lg" style={btnSecondary} onClick={onDetails}>
          Details
        </button>
      </div>
    </Card>
  );
}

/* ========================================================================= */
/* DETAIL VIEW                                                                 */
/* ========================================================================= */

const SIM_TIMEFRAMES = ['1Min', '5Min', '15Min', '1H'] as const;

function ScenariosTab({ slug, displayCode }: { slug: string; displayCode: string }) {
  const [scenarios, setScenarios] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  if (scenarios === null && !loading) {
    setLoading(true);
    import('@/lib/api/client').then(({ apiFetch }) => {
      apiFetch<any>(`/api/execution-types/${slug}/scenarios`)
        .then((data) => { setScenarios(data); setLoading(false); })
        .catch(() => { setScenarios({ scenarios: [] }); setLoading(false); });
    });
  }

  if (loading || !scenarios) {
    return <Card><p className="text-sm py-8 text-center" style={{ color: 'var(--text-muted)' }}>Loading scenarios...</p></Card>;
  }

  return (
    <div className="space-y-4">
      <Card>
        <h4 className="text-sm font-medium mb-2">Validation Scenarios</h4>
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          Deterministic test scenarios using mock data. Each scenario shows how [{displayCode}] handles a specific market condition.
        </p>
      </Card>

      {(scenarios.scenarios || []).map((scenario: any) => (
        <Card key={scenario.id}>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold px-2 py-0.5 rounded-full"
                style={{
                  color: scenario.passed ? 'var(--green)' : 'var(--red)',
                  background: (scenario.passed ? 'var(--green)' : 'var(--red)') + '20',
                }}>
                {scenario.passed ? 'PASS' : 'FAIL'}
              </span>
              <h5 className="text-sm font-medium">{scenario.name}</h5>
            </div>
            {scenario.r_multiple != null && (
              <span className="text-xs font-mono font-bold"
                style={{ color: scenario.r_multiple >= 0 ? 'var(--green)' : 'var(--red)' }}>
                {scenario.r_multiple >= 0 ? '+' : ''}{scenario.r_multiple.toFixed(1)}R
              </span>
            )}
          </div>
          <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>{scenario.description}</p>

          {/* Mini chart */}
          {scenario.chart_bars && scenario.chart_bars.length > 0 && (
            <div style={{ minHeight: 180 }}>
              <SyncedChartPane
                panes={[{
                  id: `scenario-${scenario.id}`,
                  height: 180,
                  series: [{
                    type: 'Candlestick' as const,
                    data: scenario.chart_bars.map((b: any) => ({
                      time: b.timestamp, open: b.open, high: b.high, low: b.low, close: b.close,
                    })),
                    markers: scenario.markers || [],
                    priceLines: [
                      ...(scenario.stop_price ? [{ price: scenario.stop_price, color: '#F44336', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: 'Stop' }] : []),
                      ...(scenario.target_price ? [{ price: scenario.target_price, color: '#4CAF50', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: 'Target' }] : []),
                    ],
                  }],
                }]}
              />
            </div>
          )}

          {/* Summary */}
          <div className="grid grid-cols-3 gap-3 mt-3">
            <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
              Entry: bar {scenario.entry_bar} @ ${scenario.entry_price?.toFixed(2)}
            </div>
            {scenario.exit_bar != null && (
              <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                Exit: bar {scenario.exit_bar} @ ${scenario.exit_price?.toFixed(2)}
              </div>
            )}
            {scenario.exit_reason && (
              <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                Reason: {scenario.exit_reason}
              </div>
            )}
            {scenario.confirmed !== undefined && (
              <div className="text-[10px]" style={{ color: scenario.confirmed ? 'var(--green)' : 'var(--red)' }}>
                Confirmation: {scenario.confirmed ? 'Passed' : 'Failed'} (bar {scenario.confirm_bar})
              </div>
            )}
          </div>
        </Card>
      ))}

      {scenarios.scenarios.length === 0 && (
        <Card>
          <p className="text-sm text-center py-4" style={{ color: 'var(--text-muted)' }}>No scenarios available for this execution type.</p>
        </Card>
      )}
    </div>
  );
}

function CodeBlock({ slug }: { slug: string }) {
  const [code, setCode] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  if (code === null && !loading) {
    setLoading(true);
    import('@/lib/api/client').then(({ apiFetch }) => {
      apiFetch<{ source: string }>(`/api/execution-types/${slug}/code`)
        .then((data) => { setCode(data.source); setLoading(false); })
        .catch(() => { setCode('// Failed to load code'); setLoading(false); });
    });
  }

  return (
    <div className="rounded-lg p-4 font-mono text-xs" style={{
      background: '#0d1117', color: '#c9d1d9', maxHeight: 500, overflowY: 'auto',
      lineHeight: 1.6, whiteSpace: 'pre-wrap', wordBreak: 'break-word',
    }}>
      <pre>{loading ? '# Loading...' : (code || '# No code available')}</pre>
    </div>
  );
}

function DetailView({ mod, onBack }: { mod: ExecTypeModule; onBack: () => void }) {
  const updateParams = useUpdateExecTypeParams();
  const toggleMut = useToggleExecutionType();
  const simulateMut = useSimulateExecType();
  const createVariation = useCreateVariation();

  // Simulation config
  const [simSymbol, setSimSymbol] = useState('NVDA');
  const [simTimeframe, setSimTimeframe] = useState('5Min');
  const [simDirection, setSimDirection] = useState<'LONG' | 'SHORT'>('LONG');
  const [simDays, setSimDays] = useState(7);

  return (
    <div>
      <PageHeader
        title={mod.name}
        backHref="#"
        actions={
          <div className="flex items-center gap-3">
            <Toggle enabled={mod.enabled} onToggle={() => toggleMut.mutate(mod.slug)} />
            <span className="text-xs" style={{ color: mod.enabled ? 'var(--green)' : 'var(--text-muted)' }}>
              {mod.enabled ? 'Enabled' : 'Disabled'}
            </span>
            {mod.is_default && Object.keys(mod.parameters_schema).length > 0 && (
              <button className="px-3 py-1.5 rounded-lg text-xs"
                style={{ ...btnSecondary, color: 'var(--accent)', borderColor: 'var(--accent)' }}
                onClick={() => {
                  const defaultParams: Record<string, any> = {};
                  for (const [k, v] of Object.entries(mod.parameters_schema)) {
                    defaultParams[k] = v.default;
                  }
                  const varNum = (mod.variations?.length || 0) + 1;
                  const badge = `${mod.display_code}${varNum}`;
                  createVariation.mutate({ slug: mod.slug, name: `${mod.name} (Custom)`, params: defaultParams, badge });
                }}>
                Create Variation
              </button>
            )}
            <button onClick={onBack} className="px-4 py-2 rounded-lg text-sm" style={btnSecondary}>Back</button>
          </div>
        }
      />

      {/* Badge + contexts */}
      <div className="flex items-center gap-3 mb-4 flex-wrap">
        <span className="text-xs font-mono font-bold px-2 py-1 rounded-full"
          style={{ color: EXEC_BADGE_COLOR, background: EXEC_BADGE_COLOR + '20' }}>
          [{mod.display_code}]
        </span>
        {mod.contexts.map((ctx) => {
          const colors = CONTEXT_COLORS[ctx] || { color: 'var(--text-muted)', bg: 'var(--bg-input)' };
          return (
            <span key={ctx} className="text-[9px] px-1.5 py-0.5 rounded-full capitalize"
              style={{ color: colors.color, background: colors.bg }}>
              {ctx.replace(/_/g, ' ')}
            </span>
          );
        })}
      </div>

      <TabBar tabs={['Parameters', 'Workflow Steps', 'Simulation', 'Scenarios', 'Code']}>
        {(tab) => (
          <div>
            {tab === 'Parameters' && (
              <Card>
                <div className="flex items-center justify-between mb-4">
                  <h4 className="text-sm font-medium">Parameters</h4>
                  {mod.is_default && (
                    <span className="text-[10px] px-2 py-0.5 rounded-full" style={{ color: 'var(--text-muted)', background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                      Default (create a variation to customize)
                    </span>
                  )}
                </div>
                {Object.entries(mod.parameters_schema).length === 0 ? (
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>No configurable parameters</p>
                ) : (
                  <div className="space-y-4">
                    {Object.entries(mod.parameters_schema).map(([key, schema]) => {
                      const currentValue = mod.user_params?.[key] ?? schema.default;
                      return (
                        <div key={key} className="flex items-center gap-4">
                          <div className="w-48 flex-shrink-0">
                            <p className="text-xs font-medium">{schema.label}</p>
                            <p className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>{key} ({schema.type})</p>
                          </div>
                          <div className="flex-1">
                            {schema.options ? (
                              <select value={String(currentValue)} style={inputStyle}
                                onChange={(e) => {
                                  const val = schema.type === 'int' ? parseInt(e.target.value) : e.target.value;
                                  updateParams.mutate({ slug: mod.slug, params: { ...mod.user_params, [key]: val } });
                                }}>
                                {schema.options.map((opt: any) => (
                                  <option key={String(opt)} value={String(opt)}>{String(opt)}</option>
                                ))}
                              </select>
                            ) : (
                              <input type="number" value={currentValue} min={schema.min} style={inputStyle}
                                onChange={(e) => {
                                  updateParams.mutate({ slug: mod.slug, params: { ...mod.user_params, [key]: parseInt(e.target.value) } });
                                }} />
                            )}
                          </div>
                          <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                            default: {String(schema.default)}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                )}
              </Card>
            )}

            {tab === 'Workflow Steps' && (
              <Card>
                <h4 className="text-sm font-medium mb-4">Execution Workflow</h4>
                <div className="space-y-0">
                  {mod.steps.map((step, i) => (
                    <div key={i}>
                      <div className="flex items-start gap-3">
                        <div className="flex flex-col items-center">
                          <div className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0"
                            style={{ background: 'var(--accent)', color: 'white' }}>
                            {i + 1}
                          </div>
                          {i < mod.steps.length - 1 && (
                            <div className="w-0.5 h-6" style={{ background: 'var(--border)' }} />
                          )}
                        </div>
                        <div className="pb-3">
                          <p className="text-xs font-medium" style={{ color: 'var(--text-primary)' }}>{step.label}</p>
                          <p className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>{step.action}</p>
                          {step.action === 'branch' && step.if_confirmed && (
                            <div className="mt-2 ml-2 space-y-2">
                              <div className="rounded px-2 py-1" style={{ background: 'var(--green-muted)', borderLeft: '2px solid var(--green)' }}>
                                <p className="text-[10px] font-medium" style={{ color: 'var(--green)' }}>If confirmed:</p>
                                {(step.if_confirmed as any[]).map((s: any, j: number) => (
                                  <p key={j} className="text-[10px] ml-2" style={{ color: 'var(--text-muted)' }}>{s.label}</p>
                                ))}
                              </div>
                              <div className="rounded px-2 py-1" style={{ background: 'var(--red-muted)', borderLeft: '2px solid var(--red)' }}>
                                <p className="text-[10px] font-medium" style={{ color: 'var(--red)' }}>If NOT confirmed:</p>
                                {(step.if_not_confirmed as any[]).map((s: any, j: number) => (
                                  <p key={j} className="text-[10px] ml-2" style={{ color: 'var(--text-muted)' }}>{s.label}</p>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </Card>
            )}
            {tab === 'Simulation' && (
              <div className="space-y-4">
                {/* Config */}
                <Card>
                  <h4 className="text-sm font-medium mb-3">Simulation Config</h4>
                  <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 items-end">
                    <div>
                      <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Symbol</label>
                      <input type="text" value={simSymbol} onChange={(e) => setSimSymbol(e.target.value.toUpperCase())} style={inputStyle} />
                    </div>
                    <div>
                      <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Timeframe</label>
                      <select value={simTimeframe} onChange={(e) => setSimTimeframe(e.target.value)} style={inputStyle}>
                        {SIM_TIMEFRAMES.map((tf) => <option key={tf} value={tf}>{tf}</option>)}
                      </select>
                    </div>
                    <div>
                      <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Direction</label>
                      <select value={simDirection} onChange={(e) => setSimDirection(e.target.value as 'LONG' | 'SHORT')} style={inputStyle}>
                        <option value="LONG">LONG</option>
                        <option value="SHORT">SHORT</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Days</label>
                      <input type="number" value={simDays} min={1} max={30} onChange={(e) => setSimDays(parseInt(e.target.value) || 7)} style={inputStyle} />
                    </div>
                    <button style={{ ...btnPrimary, opacity: simulateMut.isPending ? 0.5 : 1 }}
                      disabled={simulateMut.isPending}
                      onClick={() => simulateMut.mutate({ slug: mod.slug, params: { symbol: simSymbol, timeframe: simTimeframe, direction: simDirection, days: simDays } })}>
                      {simulateMut.isPending ? 'Running...' : 'Run Simulation'}
                    </button>
                  </div>
                </Card>

                {/* Error */}
                {simulateMut.isError && (
                  <Card>
                    <p className="text-sm" style={{ color: 'var(--red)' }}>
                      Simulation failed: {simulateMut.error instanceof Error ? simulateMut.error.message : 'Unknown error'}
                    </p>
                  </Card>
                )}

                {/* Results */}
                {simulateMut.data && (
                  <Card>
                    <div className="flex items-center justify-between mb-4">
                      <h4 className="text-sm font-medium">Execution Trace</h4>
                      <div className="flex items-center gap-3 text-xs" style={{ color: 'var(--text-muted)' }}>
                        <span>{simulateMut.data.symbol}</span>
                        <span>{simulateMut.data.direction}</span>
                        <span>{simulateMut.data.timeframe}</span>
                        <span className={simulateMut.data.confirmed ? '' : ''} style={{ color: simulateMut.data.confirmed ? 'var(--green)' : 'var(--red)' }}>
                          {simulateMut.data.confirmed ? 'Confirmed' : 'Not Confirmed (Bail)'}
                        </span>
                      </div>
                    </div>

                    {/* Summary strip */}
                    <div className="grid grid-cols-4 gap-3 mb-4">
                      <div className="rounded-lg px-3 py-2" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                        <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Fill Price</p>
                        <p className="text-sm font-mono font-bold">${simulateMut.data.fill_price?.toFixed(2)}</p>
                      </div>
                      {simulateMut.data.stop_price && (
                        <div className="rounded-lg px-3 py-2" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                          <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Stop</p>
                          <p className="text-sm font-mono font-bold" style={{ color: 'var(--red)' }}>${simulateMut.data.stop_price?.toFixed(2)}</p>
                        </div>
                      )}
                      {simulateMut.data.target_price && (
                        <div className="rounded-lg px-3 py-2" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                          <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Target</p>
                          <p className="text-sm font-mono font-bold" style={{ color: 'var(--green)' }}>${simulateMut.data.target_price?.toFixed(2)}</p>
                        </div>
                      )}
                      <div className="rounded-lg px-3 py-2" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                        <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Trigger</p>
                        <p className="text-sm font-mono">{simulateMut.data.trigger_time?.substring(11, 19)}</p>
                      </div>
                    </div>

                    {/* Mini chart */}
                    {simulateMut.data.chart_bars && simulateMut.data.chart_bars.length > 0 && (
                      <div className="mb-4" style={{ minHeight: 250 }}>
                        <SyncedChartPane
                          panes={[{
                            id: 'sim-price',
                            height: 250,
                            series: [{
                              type: 'Candlestick' as const,
                              data: simulateMut.data.chart_bars.map((b) => ({
                                time: b.timestamp, open: b.open, high: b.high, low: b.low, close: b.close,
                              })),
                              markers: simulateMut.data.chart_markers || [],
                              priceLines: [
                                ...(simulateMut.data.stop_price ? [{
                                  price: simulateMut.data.stop_price,
                                  color: '#F44336', lineWidth: 1, lineStyle: 2,
                                  axisLabelVisible: true, title: 'Stop',
                                }] : []),
                                ...(simulateMut.data.target_price ? [{
                                  price: simulateMut.data.target_price,
                                  color: '#4CAF50', lineWidth: 1, lineStyle: 2,
                                  axisLabelVisible: true, title: 'Target',
                                }] : []),
                              ],
                            }],
                          }]}
                        />
                      </div>
                    )}

                    {/* Step-by-step trace */}
                    <div className="space-y-0">
                      {simulateMut.data.steps.map((step, i) => {
                        const isWebhook = step.action === 'fire_webhook';
                        const isExit = step.action === 'exit';
                        const isBail = step.action === 'bail';
                        const isConfirm = step.action === 'confirmation_check';
                        const stepColor = isWebhook ? 'var(--accent)' : isBail ? 'var(--red)' : isExit ? (step.r_multiple != null && step.r_multiple >= 0 ? 'var(--green)' : 'var(--red)') : isConfirm ? (step.confirmed ? 'var(--green)' : 'var(--red)') : 'var(--text-primary)';

                        return (
                          <div key={i} className="flex items-start gap-3">
                            <div className="flex flex-col items-center">
                              <div className="w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold flex-shrink-0"
                                style={{ background: isWebhook ? 'var(--accent)' : 'var(--bg-input)', color: isWebhook ? 'white' : 'var(--text-muted)', border: isWebhook ? 'none' : '1px solid var(--border)' }}>
                                {i + 1}
                              </div>
                              {i < simulateMut.data!.steps.length - 1 && (
                                <div className="w-0.5 h-4" style={{ background: 'var(--border)' }} />
                              )}
                            </div>
                            <div className="pb-2">
                              <p className="text-xs" style={{ color: stepColor }}>{step.label}</p>
                              <div className="flex gap-2 mt-0.5">
                                <span className="text-[9px] font-mono" style={{ color: 'var(--text-muted)' }}>{step.action}</span>
                                {step.time && <span className="text-[9px] font-mono" style={{ color: 'var(--text-muted)' }}>bar {step.bar}</span>}
                                {step.price != null && <span className="text-[9px] font-mono" style={{ color: 'var(--text-muted)' }}>${step.price.toFixed(2)}</span>}
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </Card>
                )}

                {/* Empty state */}
                {!simulateMut.data && !simulateMut.isPending && !simulateMut.isError && (
                  <Card>
                    <div className="text-center py-8">
                      <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>Click <strong>Run Simulation</strong> to test this execution type on real market data.</p>
                      <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                        The simulation finds a trigger signal and traces the [{mod.display_code}] execution process step by step, including webhook events.
                      </p>
                    </div>
                  </Card>
                )}
              </div>
            )}

            {tab === 'Scenarios' && <ScenariosTab slug={mod.slug} displayCode={mod.display_code} />}

            {tab === 'Code' && (
              <Card>
                <h4 className="text-sm font-medium mb-3">Execution Type Implementation</h4>
                <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
                  This is the Python class that defines how [{mod.display_code}] {mod.name} executes trades.
                  It determines entry detection, fill price, confirmation behavior, and bail logic.
                </p>
                <CodeBlock slug={mod.slug} />
              </Card>
            )}
          </div>
        )}
      </TabBar>
    </div>
  );
}

/* ========================================================================= */
/* MAIN COMPONENT                                                              */
/* ========================================================================= */

export default function ExecutionTypesPage() {
  const { data: modules, isLoading } = useExecutionTypes();
  const toggleMut = useToggleExecutionType();
  const [detailSlug, setDetailSlug] = useState<string | null>(null);

  const detailModule = modules?.find((m) => m.slug === detailSlug);

  if (isLoading) {
    return (
      <div>
        <PageHeader title="Execution Types" subtitle="Loading..." />
      </div>
    );
  }

  // Detail view
  if (detailModule) {
    return <DetailView mod={detailModule} onBack={() => setDetailSlug(null)} />;
  }

  // List view
  return (
    <div>
      <PageHeader title="Execution Types" subtitle="How trades are entered and managed after a trigger fires. Enabled types are applied to all confluence pack triggers." />

      <div className="space-y-3">
        {(modules || []).map((mod) => (
          <div key={mod.slug}>
            <ExecCard
              mod={mod}
              onToggle={() => toggleMut.mutate(mod.slug)}
              onDetails={() => setDetailSlug(mod.slug)}
            />
            {/* Nested variations */}
            {mod.variations && mod.variations.length > 0 && (
              <div className="ml-8 mt-1 space-y-1">
                {mod.variations.map((v) => (
                  <div key={v.id} className="flex items-center gap-3 px-3 py-2 rounded-lg"
                    style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderLeft: '3px solid var(--accent)' }}>
                    <span className="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded-full"
                      style={{ color: EXEC_BADGE_COLOR, background: EXEC_BADGE_COLOR + '20' }}>
                      [{v.badge || mod.display_code}]
                    </span>
                    <span className="text-xs font-medium flex-1">{v.name}</span>
                    <span className="text-[9px] font-mono" style={{ color: 'var(--text-muted)' }}>
                      {Object.entries(v.params).map(([k, val]) => `${k}=${val}`).join(', ') || 'default params'}
                    </span>
                    <span className={`text-[9px] px-1.5 py-0.5 rounded-full`}
                      style={{ color: v.enabled ? 'var(--green)' : 'var(--text-muted)', background: v.enabled ? 'var(--green-muted)' : 'var(--bg-input)' }}>
                      {v.enabled ? 'Enabled' : 'Disabled'}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {(!modules || modules.length === 0) && (
        <Card>
          <div className="text-center py-12">
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No execution types found.</p>
          </div>
        </Card>
      )}
    </div>
  );
}
