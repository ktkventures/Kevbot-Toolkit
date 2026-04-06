'use client';

import { useState, useEffect } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import { apiFetch } from '@/lib/api/client';

/* ========================================================================= */
/* TYPES                                                                       */
/* ========================================================================= */

interface ExecTypeModule {
  slug: string;
  name: string;
  description: string;
  exec_type_codes: string[];
  contexts: string[];
  steps: Array<{
    action: string;
    label: string;
    [key: string]: any;
  }>;
  parameters_schema: Record<string, {
    type: string;
    default: any;
    options?: any[];
    label: string;
    min?: number;
  }>;
}

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

const STEP_ICONS: Record<string, string> = {
  check_trigger: '1',
  check_level_cross: '1',
  check_signal: '1',
  fill: '2',
  plot_marker: '3',
  fire_webhook: '4',
  wait_for_confirmation: '3',
  branch: '4',
  bail: '5',
  set_position: '3',
};

/* ========================================================================= */
/* COMPONENT                                                                   */
/* ========================================================================= */

export default function ExecutionTypesPage() {
  const [modules, setModules] = useState<ExecTypeModule[]>([]);
  const [selectedSlug, setSelectedSlug] = useState<string | null>(null);

  useEffect(() => {
    apiFetch<ExecTypeModule[]>('/api/execution-types')
      .then(setModules)
      .catch(() => {});
  }, []);

  const selected = modules.find((m) => m.slug === selectedSlug);

  return (
    <div>
      <PageHeader title="Execution Types" subtitle="How trades are entered and managed after a trigger fires" />

      {/* Module cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {modules.map((mod) => (
          <Card key={mod.slug}>
            <div className="cursor-pointer" onClick={() => setSelectedSlug(selectedSlug === mod.slug ? null : mod.slug)}>
              <div className="flex items-center justify-between mb-2">
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
              <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>{mod.description}</p>
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
            </div>
          </Card>
        ))}
      </div>

      {/* Detail view */}
      {selected && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Workflow Steps */}
          <Card>
            <h4 className="text-sm font-medium mb-4">Execution Workflow</h4>
            <div className="space-y-0">
              {selected.steps.map((step, i) => (
                <div key={i}>
                  {/* Step */}
                  <div className="flex items-start gap-3">
                    {/* Step number + line */}
                    <div className="flex flex-col items-center">
                      <div className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0"
                        style={{ background: 'var(--accent)', color: 'white' }}>
                        {i + 1}
                      </div>
                      {i < selected.steps.length - 1 && (
                        <div className="w-0.5 h-6" style={{ background: 'var(--border)' }} />
                      )}
                    </div>
                    {/* Step content */}
                    <div className="pb-3">
                      <p className="text-xs font-medium" style={{ color: 'var(--text-primary)' }}>{step.label}</p>
                      <p className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>{step.action}</p>
                      {/* Branch rendering */}
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

          {/* Parameters */}
          <Card>
            <h4 className="text-sm font-medium mb-4">Parameters</h4>
            {Object.entries(selected.parameters_schema).length === 0 ? (
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>No configurable parameters</p>
            ) : (
              <div className="space-y-3">
                {Object.entries(selected.parameters_schema).map(([key, schema]) => (
                  <div key={key} className="flex items-center justify-between rounded-lg px-3 py-2"
                    style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                    <div>
                      <p className="text-xs font-medium">{schema.label}</p>
                      <p className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>{key} ({schema.type})</p>
                    </div>
                    <div className="text-right">
                      <p className="text-xs font-mono font-bold" style={{ color: 'var(--accent)' }}>
                        {String(schema.default)}
                      </p>
                      {schema.options && (
                        <p className="text-[9px]" style={{ color: 'var(--text-muted)' }}>
                          Options: {schema.options.join(', ')}
                        </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Exec type codes reference */}
            <div className="mt-6">
              <h5 className="text-xs font-medium mb-2" style={{ color: 'var(--text-muted)' }}>Trigger Suffixes</h5>
              <div className="space-y-1">
                {selected.exec_type_codes.map((code) => {
                  const suffixMap: Record<string, string> = {
                    'C': '(no suffix) — bar close trigger',
                    'L0': '_ib — intra-bar level cross (current bar level)',
                    'L1': '_ib — intra-bar level cross (previous bar level)',
                    'LC': '_lc — level cross with confirmation',
                    'CC': '_cc — close-to-close with confirmation',
                    'HM': '_hm — hybrid market (legacy, use LC)',
                    'HL': '_hl — hybrid limit (legacy, use LC)',
                  };
                  return (
                    <div key={code} className="flex items-center gap-2">
                      <span className="text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded-full"
                        style={{ color: EXEC_BADGE_COLOR, background: EXEC_BADGE_COLOR + '20' }}>
                        [{code}]
                      </span>
                      <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                        {suffixMap[code] || code}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </Card>
        </div>
      )}

      {modules.length === 0 && (
        <Card>
          <div className="text-center py-12">
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Loading execution types...</p>
          </div>
        </Card>
      )}
    </div>
  );
}
