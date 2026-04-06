'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import { useExecutionTypes, useToggleExecutionType, useUpdateExecTypeParams, type ExecTypeModule } from '@/hooks/queries/useExecutionTypes';

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
          <h3 className="text-sm font-semibold">{mod.name}</h3>
          <div className="flex gap-1">
            {mod.exec_type_codes.map((code) => (
              <span key={code} className="text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded-full"
                style={{ color: EXEC_BADGE_COLOR, background: EXEC_BADGE_COLOR + '20' }}>
                [{code}]
              </span>
            ))}
          </div>
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

function DetailView({ mod, onBack }: { mod: ExecTypeModule; onBack: () => void }) {
  const updateParams = useUpdateExecTypeParams();
  const toggleMut = useToggleExecutionType();

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
            <button onClick={onBack} className="px-4 py-2 rounded-lg text-sm" style={btnSecondary}>Back</button>
          </div>
        }
      />

      {/* Exec type badges + contexts */}
      <div className="flex items-center gap-3 mb-4 flex-wrap">
        {mod.exec_type_codes.map((code) => (
          <span key={code} className="text-[10px] font-mono font-semibold px-2 py-1 rounded-full"
            style={{ color: EXEC_BADGE_COLOR, background: EXEC_BADGE_COLOR + '20' }}>
            [{code}]
          </span>
        ))}
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

      <TabBar tabs={['Parameters', 'Workflow Steps']}>
        {(tab) => (
          <div>
            {tab === 'Parameters' && (
              <Card>
                <h4 className="text-sm font-medium mb-4">Parameters</h4>
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

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {(modules || []).map((mod) => (
          <ExecCard
            key={mod.slug}
            mod={mod}
            onToggle={() => toggleMut.mutate(mod.slug)}
            onDetails={() => setDetailSlug(mod.slug)}
          />
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
