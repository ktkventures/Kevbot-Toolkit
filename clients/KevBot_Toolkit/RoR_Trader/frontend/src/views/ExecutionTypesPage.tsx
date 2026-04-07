'use client';

import { useState } from 'react';
import dynamic from 'next/dynamic';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';

const SyncedChartPane = dynamic(() => import('@/charts/SyncedChartPane'), { ssr: false });
const SandboxPanel = dynamic(() => import('@/components/SandboxPanel'), { ssr: false });
import { useExecutionTypes, useToggleExecutionType, useUpdateExecTypeParams, useCreateVariation, useDeleteVariation, type ExecTypeModule, type ExecTypeVariation } from '@/hooks/queries/useExecutionTypes';

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
    return <Card><p className="text-sm py-8 text-center" style={{ color: 'var(--text-muted)' }}>Loading scenarios from real market data...</p></Card>;
  }

  const EXEC_BADGE_COLOR_LOCAL = '#2196F3';

  return (
    <div className="space-y-4">
      <Card>
        <h4 className="text-sm font-medium mb-2">Scenario Examples</h4>
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          Real trades from NVDA 5Min showing how [{displayCode}] handles different exit conditions. Chart (with EMA overlay for demonstration) on the left, execution workflow on the right. Entry and exit drill-down charts shown below.
        </p>
      </Card>

      {(scenarios.scenarios || []).map((scenario: any) => (
        <Card key={scenario.id}>
          {/* Header */}
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <h5 className="text-sm font-medium">{scenario.name}</h5>
              {scenario.r_multiple != null && (
                <span className="text-xs font-mono font-bold"
                  style={{ color: scenario.r_multiple >= 0 ? 'var(--green)' : 'var(--red)' }}>
                  {scenario.r_multiple >= 0 ? '+' : ''}{scenario.r_multiple.toFixed(1)}R
                </span>
              )}
            </div>
            <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{scenario.direction}</span>
          </div>
          <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>{scenario.description}</p>

          {/* Side-by-side: Chart (60%) + Workflow (40%) */}
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
            {/* Left: Chart with indicator overlay */}
            <div className="lg:col-span-3">
              {scenario.chart_bars && scenario.chart_bars.length > 0 && (
                <div style={{ minHeight: 220 }}>
                  <SyncedChartPane
                    panes={[{
                      id: `scenario-${scenario.id}`,
                      height: 220,
                      series: [
                        {
                          type: 'Candlestick' as const,
                          data: scenario.chart_bars.map((b: any) => ({
                            time: b.timestamp, open: b.open, high: b.high, low: b.low, close: b.close,
                          })),
                          markers: scenario.markers || [],
                          priceLines: [
                            ...(scenario.stop_price ? [{ price: scenario.stop_price, color: '#F44336', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: 'Stop' }] : []),
                            ...(scenario.target_price ? [{ price: scenario.target_price, color: '#4CAF50', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: 'Target' }] : []),
                          ],
                        },
                        // EMA overlay
                        ...(scenario.ema_data && scenario.ema_data.length > 0 ? [{
                          type: 'Line' as const,
                          data: scenario.ema_data,
                          options: { color: '#FF9800', lineWidth: 2, title: scenario.ema_label || 'EMA' },
                        }] : []),
                      ],
                    }]}
                  />
                </div>
              )}

              {/* Drill-down charts: entry + exit */}
              <div className="grid grid-cols-2 gap-2 mt-2">
                {scenario.entry_drill && scenario.entry_drill.length > 0 && (
                  <div>
                    <p className="text-[9px] font-medium mb-1" style={{ color: 'var(--text-muted)' }}>Entry Drill-Down</p>
                    <div style={{ minHeight: 120 }}>
                      <SyncedChartPane
                        panes={[{
                          id: `entry-drill-${scenario.id}`,
                          height: 120,
                          series: [{
                            type: 'Candlestick' as const,
                            data: scenario.entry_drill.map((b: any) => ({
                              time: b.timestamp, open: b.open, high: b.high, low: b.low, close: b.close,
                            })),
                            markers: scenario.entry_markers || [],
                          }],
                        }]}
                      />
                    </div>
                  </div>
                )}
                {scenario.exit_drill && scenario.exit_drill.length > 0 && (
                  <div>
                    <p className="text-[9px] font-medium mb-1" style={{ color: 'var(--text-muted)' }}>Exit Drill-Down</p>
                    <div style={{ minHeight: 120 }}>
                      <SyncedChartPane
                        panes={[{
                          id: `exit-drill-${scenario.id}`,
                          height: 120,
                          series: [{
                            type: 'Candlestick' as const,
                            data: scenario.exit_drill.map((b: any) => ({
                              time: b.timestamp, open: b.open, high: b.high, low: b.low, close: b.close,
                            })),
                            markers: scenario.exit_markers || [],
                            priceLines: scenario.stop_price ? [{ price: scenario.stop_price, color: '#F44336', lineWidth: 1, lineStyle: 2, axisLabelVisible: false, title: '' }] : [],
                          }],
                        }]}
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Right: Workflow trace */}
            <div className="lg:col-span-2">
              <div className="rounded-lg p-3" style={{ background: 'var(--bg-primary)', border: '1px solid var(--border)' }}>
                <h6 className="text-[10px] font-medium mb-2" style={{ color: 'var(--text-muted)' }}>Execution Workflow</h6>
                <div className="space-y-0">
                  {(scenario.workflow_steps || []).map((step: any, i: number) => (
                    <div key={i} className="flex items-start gap-2">
                      <div className="flex flex-col items-center">
                        <div className="w-5 h-5 rounded-full flex items-center justify-center text-[8px] font-bold flex-shrink-0"
                          style={{
                            background: step.isWebhook ? 'var(--accent)' : step.action === 'exit' ? (step.color === 'var(--green)' ? 'var(--green)' : 'var(--red)') : 'var(--bg-input)',
                            color: step.isWebhook || step.action === 'exit' ? 'white' : 'var(--text-muted)',
                            border: step.isWebhook || step.action === 'exit' ? 'none' : '1px solid var(--border)',
                          }}>
                          {i + 1}
                        </div>
                        {i < (scenario.workflow_steps || []).length - 1 && (
                          <div className="w-0.5 h-3" style={{ background: 'var(--border)' }} />
                        )}
                      </div>
                      <div className="pb-1.5">
                        <p className="text-[10px]" style={{ color: step.color || (step.isWebhook ? 'var(--accent)' : 'var(--text-primary)') }}>
                          {step.label}
                        </p>
                        {step.badge && (
                          <span className="text-[8px] font-mono font-bold px-1 py-0.5 rounded-full"
                            style={{ color: EXEC_BADGE_COLOR_LOCAL, background: EXEC_BADGE_COLOR_LOCAL + '20' }}>
                            [{step.badge}]
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </Card>
      ))}

      {scenarios.scenarios && scenarios.scenarios.length === 0 && (
        <Card>
          <p className="text-sm text-center py-4" style={{ color: 'var(--text-muted)' }}>No scenario trades found. Try enabling more execution types or running a longer backtest period.</p>
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
  const createVariation = useCreateVariation();

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

      <TabBar tabs={['Description', 'Workflow Steps', 'Sandbox', 'Scenarios', 'Code']}>
        {(tab) => (
          <div>
            {tab === 'Description' && (
              <div className="space-y-4">
                {(() => {
                  const desc = (mod as any).detailed_description || {};
                  if (!desc.overview) {
                    return <Card><p className="text-xs" style={{ color: 'var(--text-muted)' }}>{mod.description}</p></Card>;
                  }
                  return (
                    <>
                      {/* Overview */}
                      <Card>
                        <h4 className="text-sm font-medium mb-2">Overview</h4>
                        <p className="text-xs leading-relaxed" style={{ color: 'var(--text-secondary)' }}>{desc.overview}</p>
                      </Card>

                      {/* How It Works */}
                      {desc.how_it_works && (
                        <Card>
                          <h4 className="text-sm font-medium mb-2">How It Works</h4>
                          <ol className="space-y-1.5">
                            {(desc.how_it_works as string[]).map((step: string, i: number) => (
                              <li key={i} className="flex gap-2 text-xs" style={{ color: 'var(--text-secondary)' }}>
                                <span className="text-[10px] font-bold flex-shrink-0" style={{ color: 'var(--accent)' }}>{i + 1}.</span>
                                {step}
                              </li>
                            ))}
                          </ol>
                        </Card>
                      )}

                      {/* Fill Price + Pros/Cons */}
                      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                        {desc.fill_price && (
                          <Card>
                            <h5 className="text-xs font-medium mb-1" style={{ color: 'var(--text-muted)' }}>Fill Price</h5>
                            <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>{desc.fill_price}</p>
                          </Card>
                        )}
                        {desc.pros && (
                          <Card>
                            <h5 className="text-xs font-medium mb-1" style={{ color: 'var(--green)' }}>Pros</h5>
                            <ul className="space-y-0.5">
                              {(desc.pros as string[]).map((p: string, i: number) => (
                                <li key={i} className="text-[10px]" style={{ color: 'var(--text-secondary)' }}>+ {p}</li>
                              ))}
                            </ul>
                          </Card>
                        )}
                        {desc.cons && (
                          <Card>
                            <h5 className="text-xs font-medium mb-1" style={{ color: 'var(--red)' }}>Cons</h5>
                            <ul className="space-y-0.5">
                              {(desc.cons as string[]).map((c: string, i: number) => (
                                <li key={i} className="text-[10px]" style={{ color: 'var(--text-secondary)' }}>- {c}</li>
                              ))}
                            </ul>
                          </Card>
                        )}
                      </div>

                      {/* Webhook Context */}
                      {desc.webhook_context && (
                        <Card>
                          <h4 className="text-sm font-medium mb-2">Webhook Payload Context</h4>
                          <div className="space-y-2">
                            {Object.entries(desc.webhook_context as Record<string, string>).map(([key, value]) => (
                              <div key={key} className="flex gap-3">
                                <span className="text-[10px] font-mono font-bold w-28 flex-shrink-0 px-1.5 py-0.5 rounded" style={{ color: 'var(--accent)', background: 'var(--accent-muted)' }}>
                                  {`{{${key}}}`}
                                </span>
                                <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>{value}</span>
                              </div>
                            ))}
                          </div>
                        </Card>
                      )}

                      {/* What's determined by pack vs exec type */}
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                        {desc.determined_by_pack && (
                          <Card>
                            <h5 className="text-xs font-medium mb-2" style={{ color: 'var(--text-muted)' }}>Determined by Confluence Pack</h5>
                            <ul className="space-y-1">
                              {(desc.determined_by_pack as string[]).map((item: string, i: number) => (
                                <li key={i} className="text-[10px]" style={{ color: 'var(--text-secondary)' }}>{item}</li>
                              ))}
                            </ul>
                          </Card>
                        )}
                        {desc.determined_by_exec_type && (
                          <Card>
                            <h5 className="text-xs font-medium mb-2" style={{ color: 'var(--accent)' }}>Determined by Execution Type</h5>
                            <ul className="space-y-1">
                              {(desc.determined_by_exec_type as string[]).map((item: string, i: number) => (
                                <li key={i} className="text-[10px]" style={{ color: 'var(--text-secondary)' }}>{item}</li>
                              ))}
                            </ul>
                          </Card>
                        )}
                      </div>
                      {/* Technical Specifications */}
                      {(mod as any).technical_specs && (mod as any).technical_specs.length > 0 && (
                        <Card>
                          <h4 className="text-sm font-medium mb-3">Technical Specifications</h4>
                          <div className="space-y-2">
                            {((mod as any).technical_specs as any[]).map((spec: any, i: number) => (
                              <div key={i} className="flex items-start gap-3 rounded-lg px-3 py-2"
                                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                                <span className="text-[10px] font-medium w-36 flex-shrink-0 pt-0.5" style={{ color: 'var(--text-muted)' }}>{spec.key}</span>
                                <div className="flex-1">
                                  <span className="text-xs font-mono font-bold" style={{ color: 'var(--accent)' }}>{spec.value}</span>
                                  <p className="text-[10px] mt-0.5" style={{ color: 'var(--text-muted)' }}>{spec.note}</p>
                                </div>
                              </div>
                            ))}
                          </div>
                        </Card>
                      )}
                    </>
                  );
                })()}
              </div>
            )}

            {tab === 'Workflow Steps' && (
              <div className="space-y-4">
                {(() => {
                  const CONTEXT_LABELS: Record<string, { label: string; color: string; badge: string }> = {
                    entry: { label: 'Entry Workflow', color: 'var(--green)', badge: mod.display_code },
                    exit_signal: { label: 'Exit Signal Workflow', color: 'var(--red)', badge: mod.display_code },
                    stop: { label: 'Stop Loss Workflow', color: 'var(--orange)', badge: 'L' },
                    target: { label: 'Take Profit Workflow', color: 'var(--accent)', badge: 'L' },
                  };
                  const stepsDict = (mod.steps && typeof mod.steps === 'object' && !Array.isArray(mod.steps))
                    ? mod.steps as Record<string, any[]>
                    : { entry: mod.steps as any[] || [] };

                  return Object.entries(stepsDict).map(([context, contextSteps]) => {
                    const meta = CONTEXT_LABELS[context] || { label: context, color: 'var(--text-muted)', badge: '?' };
                    return (
                      <Card key={context}>
                        <div className="flex items-center gap-2 mb-3">
                          <span className="text-xs font-mono font-bold px-1.5 py-0.5 rounded-full"
                            style={{ color: EXEC_BADGE_COLOR, background: EXEC_BADGE_COLOR + '20' }}>
                            [{meta.badge}]
                          </span>
                          <h4 className="text-sm font-medium" style={{ color: meta.color }}>{meta.label}</h4>
                        </div>
                        <div className="space-y-0">
                          {(contextSteps as any[]).map((step: any, i: number) => (
                            <div key={i}>
                              <div className="flex items-start gap-3">
                                <div className="flex flex-col items-center">
                                  <div className="w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold flex-shrink-0"
                                    style={{ background: step.action === 'fire_webhook' ? 'var(--accent)' : 'var(--bg-input)', color: step.action === 'fire_webhook' ? 'white' : 'var(--text-muted)', border: step.action === 'fire_webhook' ? 'none' : '1px solid var(--border)' }}>
                                    {i + 1}
                                  </div>
                                  {i < contextSteps.length - 1 && (
                                    <div className="w-0.5 h-4" style={{ background: 'var(--border)' }} />
                                  )}
                                </div>
                                <div className="pb-2">
                                  <p className="text-xs" style={{ color: step.action === 'fire_webhook' ? 'var(--accent)' : 'var(--text-primary)' }}>{step.label}</p>
                                  <p className="text-[9px] font-mono" style={{ color: 'var(--text-muted)' }}>{step.action}</p>
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
                    );
                  });
                })()}
              </div>
            )}
            {tab === 'Sandbox' && (
              <div className="space-y-4">
                <Card>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    Run a real backtest to verify [{mod.display_code}] execution behavior. Select any entry/exit trigger, stop/target, and confluence conditions — the same inputs as the Strategy Builder. The chart and trade results show exactly how this execution type handles each trade.
                  </p>
                </Card>
                <SandboxPanel packSlug="" layout="horizontal" />
              </div>
            )}

            {/* Old simulation removed — replaced by Backtest tab with SandboxPanel */}

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
