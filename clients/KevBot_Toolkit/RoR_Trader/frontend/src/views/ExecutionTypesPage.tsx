'use client';

import { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';

const SandboxPanel = dynamic(() => import('@/components/SandboxPanel'), { ssr: false });
const ScenarioReplayCard = dynamic(() => import('@/components/ScenarioReplayCard'), { ssr: false });
import { useExecutionTypes, useToggleExecutionType, useUpdateExecTypeParams, useCreateVariation, useDeleteVariation, useGenerateScenarios, type ExecTypeModule, type ExecTypeVariation } from '@/hooks/queries/useExecutionTypes';
import { useConfluenceTriggers, useStopLossPacks, useTakeProfitPacks } from '@/hooks/queries/usePacks';

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

const SCENARIO_CATEGORIES = [
  { key: 'common', label: 'Common' },
  { key: 'ambiguous', label: 'Ambiguous Candles' },
  { key: 'ltype', label: 'L-Type' },
  { key: 'confirmation', label: 'Confirmation (LC/CC)' },
  { key: 'edge', label: 'Edge Cases' },
] as const;

const GEN_TIMEFRAMES = ['1Min', '5Min', '15Min', '1H', '1Day'] as const;

/* ---- Static reference scenarios (Scenarios tab) ---- */
function ScenariosTab({ slug, displayCode }: { slug: string; displayCode: string }) {
  const [scenarios, setScenarios] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [activeCategory, setActiveCategory] = useState('common');

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

  const allScenarios = scenarios.scenarios || [];
  const filtered = allScenarios.filter((s: any) => (s.category || 'common') === activeCategory);
  const availableCategories = SCENARIO_CATEGORIES.filter(
    cat => allScenarios.some((s: any) => (s.category || 'common') === cat.key)
  );

  return (
    <div className="space-y-4">
      <Card>
        <h4 className="text-sm font-medium mb-2">Reference Scenarios</h4>
        <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
          Pre-curated trades from NVDA 5Min showing how [{displayCode}] handles different exit conditions. Use replay controls to step through each trade second by second.
        </p>
        <div className="flex gap-2 flex-wrap">
          {availableCategories.map(cat => (
            <button key={cat.key} onClick={() => setActiveCategory(cat.key)}
              className="text-[10px] font-medium px-3 py-1 rounded-full transition-colors"
              style={{
                background: activeCategory === cat.key ? 'var(--accent)' : 'var(--bg-input)',
                color: activeCategory === cat.key ? 'white' : 'var(--text-muted)',
                border: `1px solid ${activeCategory === cat.key ? 'var(--accent)' : 'var(--border)'}`,
                cursor: 'pointer',
              }}>
              {cat.label}
            </button>
          ))}
          {SCENARIO_CATEGORIES.filter(cat => !availableCategories.some(a => a.key === cat.key)).map(cat => (
            <span key={cat.key} className="text-[10px] font-medium px-3 py-1 rounded-full"
              style={{ background: 'var(--bg-input)', color: 'var(--text-muted)', border: '1px solid var(--border)', opacity: 0.4 }}>
              {cat.label}
            </span>
          ))}
        </div>
      </Card>
      {filtered.map((scenario: any) => (
        <ScenarioReplayCard key={scenario.id} scenario={scenario} displayCode={displayCode} />
      ))}
      {filtered.length === 0 && (
        <Card><p className="text-sm text-center py-4" style={{ color: 'var(--text-muted)' }}>No scenarios in this category yet.</p></Card>
      )}
    </div>
  );
}

/* ---- Custom scenario generator (Custom Scenarios tab) ---- */
function CustomScenariosTab({ slug, displayCode }: { slug: string; displayCode: string }) {
  const [genSymbol, setGenSymbol] = useState('NVDA');
  const [genTimeframe, setGenTimeframe] = useState('5Min');
  const [genDirection, setGenDirection] = useState<'LONG' | 'SHORT'>('LONG');
  const [genDays, setGenDays] = useState(30);
  const [genEntryTrigger, setGenEntryTrigger] = useState('');
  const [genExitTrigger, setGenExitTrigger] = useState('');
  const [genStopPack, setGenStopPack] = useState('');
  const [genTargetPack, setGenTargetPack] = useState('');
  const [genBarCount, setGenBarCount] = useState(8);
  const [customScenarios, setCustomScenarios] = useState<any>(null);

  const generateMut = useGenerateScenarios(slug);
  const { data: entryTriggers } = useConfluenceTriggers(genDirection);
  const { data: exitTriggers } = useConfluenceTriggers('EXIT');
  const { data: stopPacks } = useStopLossPacks();
  const { data: targetPacks } = useTakeProfitPacks();

  const entryOptions = useMemo(() => {
    if (!entryTriggers) return [];
    return Object.entries(entryTriggers).map(([id, name]) => ({ id, name: String(name) }));
  }, [entryTriggers]);

  const exitOptions = useMemo(() => {
    if (!exitTriggers) return [];
    return Object.entries(exitTriggers).map(([id, name]) => ({ id, name: String(name) }));
  }, [exitTriggers]);

  const stopPackList = useMemo(() => {
    if (!stopPacks) return [];
    return (stopPacks as any[]).map((p) => ({ id: p.id, label: `${p.base_template} (${p.version})` }));
  }, [stopPacks]);

  const targetPackList = useMemo(() => {
    if (!targetPacks) return [];
    return (targetPacks as any[]).map((p) => ({ id: p.id, label: `${p.base_template} (${p.version})` }));
  }, [targetPacks]);

  const handleGenerate = () => {
    if (!genEntryTrigger) return;
    generateMut.mutate({
      symbol: genSymbol,
      timeframe: genTimeframe,
      direction: genDirection,
      days: genDays,
      entry_trigger_confluence_id: genEntryTrigger,
      exit_trigger_confluence_ids: genExitTrigger ? [genExitTrigger] : [],
      stop_loss_pack_id: genStopPack || undefined,
      take_profit_pack_id: genTargetPack || undefined,
      bar_count_exit: genBarCount || null,
    } as any, {
      onSuccess: (data: any) => setCustomScenarios(data),
    });
  };

  const customList = customScenarios?.scenarios || [];

  return (
    <div className="space-y-4">
      <Card>
        <h4 className="text-sm font-medium mb-2">Generate Custom Scenarios</h4>
        <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
          Select triggers and configuration to generate scenario examples from real backtest data. The system cherry-picks representative trades for each exit type (stop, target, signal, bar count).
        </p>
        {/* Config row 1: Symbol, Timeframe, Direction, Days */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-2">
          <div>
            <label className="text-[9px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Symbol</label>
            <input value={genSymbol} onChange={e => setGenSymbol(e.target.value.toUpperCase())} style={inputStyle} />
          </div>
          <div>
            <label className="text-[9px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Timeframe</label>
            <select value={genTimeframe} onChange={e => setGenTimeframe(e.target.value)} style={inputStyle}>
              {GEN_TIMEFRAMES.map(tf => <option key={tf} value={tf}>{tf}</option>)}
            </select>
          </div>
          <div>
            <label className="text-[9px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Direction</label>
            <select value={genDirection} onChange={e => setGenDirection(e.target.value as 'LONG' | 'SHORT')} style={inputStyle}>
              <option value="LONG">LONG</option>
              <option value="SHORT">SHORT</option>
            </select>
          </div>
          <div>
            <label className="text-[9px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Days</label>
            <input type="number" value={genDays} onChange={e => setGenDays(Number(e.target.value))} min={5} max={90} style={inputStyle} />
          </div>
        </div>
        {/* Config row 2: Entry/Exit triggers */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mb-2">
          <div>
            <label className="text-[9px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Entry Trigger</label>
            <select value={genEntryTrigger} onChange={e => setGenEntryTrigger(e.target.value)} style={inputStyle}>
              <option value="">Select entry trigger...</option>
              {entryOptions.map(t => <option key={t.id} value={t.id}>{t.name}</option>)}
            </select>
          </div>
          <div>
            <label className="text-[9px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Exit Trigger</label>
            <select value={genExitTrigger} onChange={e => setGenExitTrigger(e.target.value)} style={inputStyle}>
              <option value="">(none — stop/target/bar count only)</option>
              {exitOptions.map(t => <option key={t.id} value={t.id}>{t.name}</option>)}
            </select>
          </div>
        </div>
        {/* Config row 3: Stop/Target packs + bar count */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 mb-3">
          <div>
            <label className="text-[9px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Stop Loss</label>
            <select value={genStopPack} onChange={e => setGenStopPack(e.target.value)} style={inputStyle}>
              <option value="">None</option>
              {stopPackList.map(p => <option key={p.id} value={p.id}>{p.label}</option>)}
            </select>
          </div>
          <div>
            <label className="text-[9px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Take Profit</label>
            <select value={genTargetPack} onChange={e => setGenTargetPack(e.target.value)} style={inputStyle}>
              <option value="">None</option>
              {targetPackList.map(p => <option key={p.id} value={p.id}>{p.label}</option>)}
            </select>
          </div>
          <div>
            <label className="text-[9px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Bar Count Exit</label>
            <input type="number" value={genBarCount} onChange={e => setGenBarCount(Number(e.target.value))} min={0} style={inputStyle} />
          </div>
        </div>
        <button
          onClick={handleGenerate}
          disabled={!genEntryTrigger || generateMut.isPending}
          className="text-xs font-medium px-4 py-1.5 rounded-lg transition-colors"
          style={{
            background: genEntryTrigger && !generateMut.isPending ? 'var(--accent)' : 'var(--bg-input)',
            color: genEntryTrigger && !generateMut.isPending ? 'white' : 'var(--text-muted)',
            border: `1px solid ${genEntryTrigger ? 'var(--accent)' : 'var(--border)'}`,
            cursor: genEntryTrigger && !generateMut.isPending ? 'pointer' : 'default',
            opacity: generateMut.isPending ? 0.6 : 1,
          }}
        >
          {generateMut.isPending ? 'Generating scenarios...' : 'Generate Scenarios'}
        </button>
      </Card>

      {/* Results */}
      {generateMut.isPending && (
        <Card><p className="text-sm py-6 text-center" style={{ color: 'var(--text-muted)' }}>Running backtest and fetching 1-second bars... This may take 5-15 seconds.</p></Card>
      )}
      {generateMut.isError && (
        <Card><p className="text-sm py-4 text-center" style={{ color: 'var(--red)' }}>Failed to generate scenarios. {(generateMut.error as any)?.message || 'Check API connection.'}</p></Card>
      )}
      {customList.length > 0 && (
        <>
          <div className="flex items-center gap-2 px-1">
            <h5 className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Generated Scenarios ({customList.length})</h5>
            <span className="text-[9px]" style={{ color: 'var(--text-muted)' }}>{genSymbol} {genTimeframe} {genDirection}</span>
          </div>
          {customList.map((scenario: any, i: number) => (
            <ScenarioReplayCard key={`gen-${scenario.id}-${i}`} scenario={scenario} displayCode={displayCode} />
          ))}
        </>
      )}
      {customScenarios && customList.length === 0 && !generateMut.isPending && (
        <Card><p className="text-sm text-center py-4" style={{ color: 'var(--text-muted)' }}>No trades found for this configuration. Try a longer lookback period or different triggers.</p></Card>
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

      <TabBar tabs={['Description', 'Workflow Steps', 'Sandbox', 'Scenarios', 'Custom Scenarios', 'Code']}>
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

            {tab === 'Custom Scenarios' && <CustomScenariosTab slug={mod.slug} displayCode={mod.display_code} />}

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
