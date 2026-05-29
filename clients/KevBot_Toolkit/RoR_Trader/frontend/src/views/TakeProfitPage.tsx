'use client';

/**
 * Take Profit Packs — Faithful copy of V1 design with mock data replaced by API hooks.
 * Source: src/app/confluence-packs/take-profit/versions/V1.tsx (515 lines)
 *
 * Data convention:
 *   Real value = live from API
 *   -- = wired but no data yet
 *   {{field}} = not wired, needs backend work
 */

import { useState, useMemo, useEffect } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import ChartPlaceholder from '@/components/ChartPlaceholder';
import Link from 'next/link';
import { useRiskManagementPacks, useRiskManagementTemplates } from '@/hooks/queries/usePacks';
import type { RiskManagementPackDTO, RMTemplateDTO } from '@/hooks/queries/usePacks';
import { useSaveRiskManagementPacks } from '@/hooks/mutations/usePackMutations';

/* ========================================================================
   Types
   ======================================================================== */

interface PackParam {
  key: string;
  label: string;
  type: 'int' | 'float';
  value: number;
  default: number;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
}

interface TargetPack {
  id: string;
  templateKey: string;
  name: string;
  version: string;
  tags: string[];
  enabled: boolean;
  isDefault: boolean;
  isSaved: boolean;
  description: string;
  params: PackParam[];
}

/* ========================================================================
   Target-related template keys — used to filter risk management templates
   ======================================================================== */

// Backend template keys from risk_management_packs.py TEMPLATES dict
const TARGET_TEMPLATE_KEYS = [
  'rr_ratio', 'atr_based', 'fixed_dollar', 'percentage', 'swing',
  'atr_trailing', 'breakeven_stop',
  // Legacy frontend aliases (keep for backwards compatibility)
  'risk_reward', 'atr_target', 'fixed_dollar_target', 'percentage_target', 'swing_target', 'none',
];

/* ========================================================================
   API → V1 TargetPack Mapping
   ======================================================================== */

function apiToTargetPack(dto: RiskManagementPackDTO, template?: RMTemplateDTO): TargetPack {
  const schema = template?.parameters_schema ?? {};
  // Only show target-relevant params (filter by target_params if available)
  const targetKeys = (template as any)?.target_params as string[] | undefined;
  const filteredEntries = targetKeys
    ? Object.entries(schema).filter(([key]) => targetKeys.includes(key))
    : Object.entries(schema);
  const params: PackParam[] = filteredEntries.map(([key, s]) => ({
    key,
    label: s.label || key,
    type: (s.type === 'int' ? 'int' : 'float') as 'int' | 'float',
    value: (dto.parameters?.[key] ?? s.default ?? 0) as number,
    default: (s.default ?? 0) as number,
  }));

  // Derive tags from template category
  const tags: string[] = template?.category ? [template.category] : [];

  return {
    id: dto.id,
    templateKey: dto.base_template,
    name: template?.name ?? dto.base_template,
    version: dto.version,
    tags,
    enabled: dto.enabled,
    isDefault: dto.is_default,
    isSaved: true,
    description: template?.description ?? dto.description,
    params,
  };
}

/* ========================================================================
   TargetPack → RiskManagementPackDTO (reverse mapper for save)
   ======================================================================== */

function targetPackToDto(pack: TargetPack): RiskManagementPackDTO {
  const parameters: Record<string, unknown> = {};
  for (const p of pack.params) {
    parameters[p.key] = p.value;
  }
  return {
    id: pack.id,
    base_template: pack.templateKey,
    version: pack.version,
    description: pack.description,
    enabled: pack.enabled,
    is_default: pack.isDefault,
    parameters,
  };
}

/** Build a summary string from pack parameters (replaces targetSummary functions) */
function buildParamSummary(params: PackParam[]): string {
  if (params.length === 0) return 'No target -- exit via stop, signal, or bar count';
  return params.map((p) => `${p.label}: ${p.value}`).join(', ');
}

/* ========================================================================
   Shared Components
   ======================================================================== */

const tagColors: Record<string, { color: string; bg: string }> = {
  Composite: { color: 'var(--accent)', bg: 'var(--accent-muted)' },
  Volatility: { color: 'var(--orange)', bg: 'var(--orange-muted)' },
  Fixed: { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  Structure: { color: 'var(--green)', bg: 'var(--green-muted)' },
  None: { color: 'var(--text-muted)', bg: 'var(--bg-input)' },
};

function TagBadge({ tag }: { tag: string }) {
  const s = tagColors[tag] || { color: 'var(--text-muted)', bg: 'var(--bg-input)' };
  return <span className="text-[10px] font-medium px-1.5 py-0.5 rounded" style={{ color: s.color, background: s.bg }}>{tag}</span>;
}

function Toggle({ enabled, onChange }: { enabled: boolean; onChange: () => void }) {
  return (
    <button onClick={(e) => { e.stopPropagation(); onChange(); }} className="w-9 h-5 rounded-full relative flex-shrink-0 transition-colors" style={{ background: enabled ? 'var(--accent)' : 'var(--bg-input)', border: enabled ? 'none' : '1px solid var(--border)' }}>
      <div className="w-3.5 h-3.5 rounded-full absolute transition-all" style={{ background: enabled ? 'white' : 'var(--text-muted)', top: '3px', left: enabled ? '19px' : '3px' }} />
    </button>
  );
}

/* ========================================================================
   Detail View Tabs
   ======================================================================== */

function ParametersTab({ pack, onParamChange }: { pack: TargetPack; onParamChange?: (key: string, val: number) => void }) {
  if (pack.params.length === 0) {
    return (
      <Card>
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Take Profit Parameters</h3>
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{pack.description}</p>
      </Card>
    );
  }
  return (
    <Card>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Take Profit Parameters</h3>
        {pack.isSaved && (
          <span className="text-xs px-2 py-1 rounded-lg flex items-center gap-1.5" style={{ background: 'var(--orange-muted)', color: 'var(--orange)' }}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><rect x="3" y="11" width="18" height="11" rx="2" /><path d="M7 11V7a5 5 0 0110 0v4" /></svg>
            Locked after save
          </span>
        )}
      </div>
      {pack.isSaved && (
        <div className="mb-4 px-3 py-2 rounded-lg text-xs" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>
          Parameters are locked to protect active strategies. Create a new variation to test different settings.
        </div>
      )}
      <div className="space-y-3">
        {pack.params.map((param) => (
          <div key={param.key} className="flex items-center gap-4">
            <label className="text-sm w-44 flex-shrink-0" style={{ color: 'var(--text-secondary)' }}>{param.label}</label>
            <input
              type="number"
              className="w-24 px-3 py-2 rounded-lg text-sm text-center font-mono"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: pack.isSaved ? 'var(--text-muted)' : 'var(--text-primary)', cursor: pack.isSaved ? 'not-allowed' : 'text' }}
              value={param.value}
              disabled={pack.isSaved}
              readOnly={pack.isSaved}
              min={param.min}
              max={param.max}
              step={param.step ?? (param.type === 'int' ? 1 : 0.01)}
              onChange={(e) => {
                if (pack.isSaved || !onParamChange) return;
                const v = param.type === 'int'
                  ? parseInt(e.target.value, 10)
                  : parseFloat(e.target.value);
                if (!Number.isNaN(v)) onParamChange(param.key, v);
              }}
            />
            {param.unit && <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{param.unit}</span>}
            {param.min !== undefined && <span className="text-xs" style={{ color: 'var(--text-muted)' }}>({param.min}{'\u2013'}{param.max})</span>}
            {param.value !== param.default && <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ color: 'var(--orange)', background: 'var(--orange-muted)' }}>default: {param.default}</span>}
          </div>
        ))}
      </div>
      {!pack.isSaved && (
        <div className="mt-6 px-3 py-2 rounded-lg text-xs flex items-center gap-2" style={{ background: 'var(--orange-muted)', color: 'var(--orange)' }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" /><line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" /></svg>
          Once saved, parameters cannot be changed. Use the Preview tab to test, then &quot;Save as Variation&quot; when ready.
        </div>
      )}
      <div className="mt-6 pt-4 border-t" style={{ borderColor: 'var(--border)' }}>
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          Template: <span className="font-medium" style={{ color: 'var(--text-secondary)' }}>{pack.name}</span>
          {' \u00b7 '}{pack.description}
        </p>
        <p className="text-xs mt-1 font-mono" style={{ color: 'var(--text-muted)' }}>Effective: {buildParamSummary(pack.params)}</p>
      </div>
    </Card>
  );
}

function BehaviorTab({ pack }: { pack: TargetPack }) {
  return (
    <Card>
      <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Take Profit Behavior</h3>
      <div className="space-y-4">
        <div className="px-3 py-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
          <p className="text-xs font-medium mb-1" style={{ color: 'var(--text-primary)' }}>Method: {pack.name}</p>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{pack.description}</p>
        </div>
        <div className="px-3 py-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
          <p className="text-xs font-medium mb-1" style={{ color: 'var(--text-primary)' }}>Effective Target</p>
          <p className="text-sm font-mono" style={{ color: 'var(--green)' }}>{buildParamSummary(pack.params)}</p>
        </div>
        <div className="px-3 py-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
          <p className="text-xs font-medium mb-1" style={{ color: 'var(--text-primary)' }}>Evaluation</p>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            Target is calculated on entry and checked on every subsequent bar.
            {pack.templateKey === 'risk_reward' && ' Requires a stop loss pack to determine the risk amount.'}
            {pack.templateKey === 'none' && ' No target is set \u2014 position exits only via stop loss, signal, or bar count.'}
          </p>
        </div>
        <div className="px-3 py-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
          <p className="text-xs font-medium mb-1" style={{ color: 'var(--text-primary)' }}>Exit Priority</p>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            Take profit is checked <strong>second</strong> on each bar, after stop loss. If both are breached on the same bar, the stop loss exit is used (pessimistic assumption).
          </p>
        </div>
      </div>
    </Card>
  );
}

function PreviewTab({ pack }: { pack: TargetPack }) {
  return (
    <div className="space-y-4">
      <div className="flex gap-3 mb-2">
        {[['NVDA', 'SPY', 'AAPL', 'TSLA'], ['1Min', '5Min', '15Min', '1H'], ['LONG', 'SHORT']].map((opts, i) => (
          <select key={i} className="px-3 py-2 rounded-lg text-sm" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}>
            {opts.map((o) => <option key={o}>{o}</option>)}
          </select>
        ))}
      </div>
      <Card>
        <ChartPlaceholder label={`Price chart with ${pack.name} placement \u2014 ${buildParamSummary(pack.params)}`} height={350} />
      </Card>
      <Card>
        <h4 className="text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Recent Target Examples</h4>
        <p className="text-xs py-8 text-center" style={{ color: 'var(--text-muted)' }}>
          Recent target examples will appear here once strategies are running.
        </p>
      </Card>
    </div>
  );
}

function CodeTab({ pack }: { pack: TargetPack }) {
  const [showCode, setShowCode] = useState(false);
  const codeBlockStyle: React.CSSProperties = { background: 'var(--bg-primary)', border: '1px solid var(--border)', color: 'var(--text-secondary)', fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.6', padding: '16px', borderRadius: '8px', whiteSpace: 'pre-wrap', overflowX: 'auto' };
  return (
    <div className="space-y-4">
      <Card>
        <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Active Configuration</h3>
        <div className="rounded-lg px-4 py-3" style={{ background: 'var(--bg-primary)', border: '1px solid var(--border)' }}>
          <div className="grid grid-cols-2 gap-y-2 gap-x-8">
            <div className="flex justify-between"><span className="text-xs" style={{ color: 'var(--text-muted)' }}>method</span><span className="text-xs font-mono font-semibold" style={{ color: 'var(--text-primary)' }}>{pack.templateKey}</span></div>
            {pack.params.map((p) => <div key={p.key} className="flex justify-between"><span className="text-xs" style={{ color: 'var(--text-muted)' }}>{p.key}</span><span className="text-xs font-mono font-semibold" style={{ color: 'var(--text-primary)' }}>{p.value}</span></div>)}
          </div>
        </div>
      </Card>
      <Card>
        <button className="flex items-center justify-between w-full text-left" onClick={() => setShowCode(!showCode)}>
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Target Calculation Function</h3>
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{showCode ? 'Collapse' : 'Expand'}</span>
        </button>
        {showCode && (
          <div className="mt-3"><pre style={codeBlockStyle}>
{`def calculate_target_price(entry_price, stop_price, direction, row, df, bar_index, target_config):
    """
    Calculate take profit price using "${pack.templateKey}" method.
    Config: ${JSON.stringify(Object.fromEntries(pack.params.map((p) => [p.key, p.value])))}

    Returns target price for the given entry, or None if no target.
    """
    # Implementation in triggers.py
    ...`}
          </pre></div>
        )}
      </Card>
    </div>
  );
}

function DangerZoneTab({ pack, onRename, onDelete }: { pack: TargetPack; onRename?: (newVersion: string) => void; onDelete?: () => void }) {
  const [renameValue, setRenameValue] = useState(pack.version);
  return (
    <Card>
      <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--red)' }}>Danger Zone</h3>
      <div className="space-y-6">
        <div>
          <label className="text-sm mb-2 block" style={{ color: 'var(--text-secondary)' }}>Rename Variation</label>
          <div className="flex gap-3">
            <input className="px-3 py-2 rounded-lg text-sm flex-1" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: pack.isDefault ? 'var(--text-muted)' : 'var(--text-primary)', cursor: pack.isDefault ? 'not-allowed' : 'text' }} value={renameValue} onChange={(e) => setRenameValue(e.target.value)} disabled={pack.isDefault} />
            <button className="px-4 py-2 rounded-lg text-sm" style={{ background: pack.isDefault ? 'var(--bg-input)' : 'var(--bg-card)', border: '1px solid var(--border)', color: pack.isDefault ? 'var(--text-muted)' : 'var(--text-secondary)' }} disabled={pack.isDefault} onClick={() => onRename?.(renameValue)}>Rename</button>
          </div>
          {pack.isDefault && <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>Default packs cannot be renamed.</p>}
        </div>
        <div className="pt-4 border-t" style={{ borderColor: 'var(--border)' }}>
          <p className="text-sm mb-2" style={{ color: 'var(--text-secondary)' }}>Delete this variation permanently.</p>
          <button className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: pack.isDefault ? 'var(--bg-input)' : 'var(--red)', color: pack.isDefault ? 'var(--text-muted)' : 'white' }} disabled={pack.isDefault} onClick={() => onDelete?.()}>Delete Variation</button>
          {pack.isDefault && <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>Default packs cannot be deleted.</p>}
        </div>
      </div>
    </Card>
  );
}

/* ========================================================================
   Pack Card
   ======================================================================== */

function PackCard({ pack, onToggle, onDetails, onCopy, hasVariations, isExpanded, onToggleExpand }: {
  pack: TargetPack; onToggle: () => void; onDetails: () => void; onCopy: () => void;
  hasVariations?: boolean; isExpanded?: boolean; onToggleExpand?: () => void;
}) {
  const isVariation = !pack.isDefault;
  return (
    <div className="flex items-center gap-3 px-4 py-3 rounded-xl border transition-colors" style={{ background: 'var(--bg-card)', borderColor: 'var(--border)', boxShadow: 'var(--card-shadow)', backdropFilter: 'var(--card-backdrop)', WebkitBackdropFilter: 'var(--card-backdrop)', opacity: pack.enabled ? 1 : 0.6, marginLeft: isVariation ? 24 : 0 }}>
      {hasVariations ? (
        <button onClick={(e) => { e.stopPropagation(); onToggleExpand?.(); }} className="w-5 h-5 rounded flex items-center justify-center text-xs flex-shrink-0 transition-transform" style={{ color: 'var(--text-muted)', transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)' }}>{'\u25B6'}</button>
      ) : isVariation ? <div className="w-5 flex-shrink-0" /> : null}
      <Toggle enabled={pack.enabled} onChange={onToggle} />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1 flex-wrap">
          <span className="font-semibold text-sm" style={{ color: 'var(--text-primary)' }}>{pack.name}</span>
          <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ color: pack.isDefault ? 'var(--text-muted)' : 'var(--text-secondary)', background: 'var(--bg-input)' }}>{pack.version}</span>
          {pack.tags.map((tag) => <TagBadge key={tag} tag={tag} />)}
        </div>
        <p className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>{buildParamSummary(pack.params)}</p>
      </div>
      <div className="flex gap-2 flex-shrink-0">
        <button onClick={onDetails} className="px-3 py-1.5 rounded text-xs font-medium" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>Details</button>
        <button onClick={onCopy} className="px-3 py-1.5 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>Create Variation</button>
      </div>
    </div>
  );
}

/* ========================================================================
   Main Component
   ======================================================================== */

export default function TakeProfitPage() {
  // --- API hooks (MUST come before any early returns) ---
  const { data: apiPacks, isLoading: packsLoading, error: packsError } = useRiskManagementPacks();
  const { data: apiTemplates, isLoading: templatesLoading } = useRiskManagementTemplates();
  const saveMutation = useSaveRiskManagementPacks();

  const [packs, setPacks] = useState<TargetPack[]>([]);
  const [detailPack, setDetailPack] = useState<TargetPack | null>(null);
  const [draftPack, setDraftPack] = useState<TargetPack | null>(null);
  const [search, setSearch] = useState('');
  const [expandedTemplates, setExpandedTemplates] = useState<Set<string>>(new Set());
  const [initialized, setInitialized] = useState(false);

  // Initialize packs from API data
  useEffect(() => {
    if (apiPacks && apiTemplates && !initialized) {
      const tgtPacks = apiPacks
        .filter((p) => TARGET_TEMPLATE_KEYS.includes(p.base_template) || (apiTemplates[p.base_template] as any)?.has_target)
        .map((p) => apiToTargetPack(p, apiTemplates[p.base_template]));
      setPacks(tgtPacks);
      const variationKeys = tgtPacks.filter((p) => !p.isDefault).map((p) => p.templateKey);
      setExpandedTemplates(new Set(variationKeys));
      setInitialized(true);
    }
  }, [apiPacks, apiTemplates, initialized]);

  // Derived state
  const isLoading = packsLoading || templatesLoading;

  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');

  function persistPacks(updatedPacks: TargetPack[]) {
    setSaveStatus('saving');
    saveMutation.mutate(updatedPacks.map(targetPackToDto), {
      onSuccess: () => { setSaveStatus('saved'); setTimeout(() => setSaveStatus('idle'), 2000); },
      onError: () => { setSaveStatus('error'); setTimeout(() => setSaveStatus('idle'), 3000); },
    });
  }

  const enabledCount = packs.filter((p) => p.enabled).length;
  function togglePack(id: string) {
    setPacks((prev) => {
      const updated = prev.map((p) => (p.id === id ? { ...p, enabled: !p.enabled } : p));
      persistPacks(updated);
      return updated;
    });
  }
  function toggleTemplate(key: string) { setExpandedTemplates((prev) => { const next = new Set(prev); if (next.has(key)) next.delete(key); else next.add(key); return next; }); }

  const groupedPacks = useMemo(() => {
    const groups: { templateKey: string; default: TargetPack; variations: TargetPack[] }[] = [];
    for (const p of packs) { if (p.isDefault) groups.push({ templateKey: p.templateKey, default: p, variations: [] }); }
    for (const p of packs) { if (!p.isDefault) { const g = groups.find((gr) => gr.templateKey === p.templateKey); if (g) g.variations.push(p); } }
    return groups;
  }, [packs]);

  const filteredGroups = useMemo(() => {
    if (!search.trim()) return groupedPacks;
    const q = search.toLowerCase();
    return groupedPacks.map((g) => ({ ...g, variations: g.variations.filter((v) => v.name.toLowerCase().includes(q) || v.version.toLowerCase().includes(q) || v.tags.some((t) => t.toLowerCase().includes(q))) })).filter((g) => g.default.name.toLowerCase().includes(q) || g.default.tags.some((t) => t.toLowerCase().includes(q)) || g.variations.length > 0);
  }, [groupedPacks, search]);

  // Loading / error states (after all hooks)
  if (isLoading) {
    return (
      <div>
        <PageHeader title="Take Profit Packs" subtitle="Loading..." />
        <Card><p className="text-sm py-8 text-center" style={{ color: 'var(--text-muted)' }}>Loading take profit packs...</p></Card>
      </div>
    );
  }
  if (packsError) {
    return (
      <div>
        <PageHeader title="Take Profit Packs" subtitle="Error loading data" />
        <Card><p className="text-sm py-8 text-center" style={{ color: 'var(--red)' }}>Failed to load packs: {String(packsError)}</p></Card>
      </div>
    );
  }

  const activePack = draftPack || detailPack;
  const isDraft = draftPack !== null;

  function handleCopy(source: TargetPack) {
    setDraftPack({ ...source, id: `${source.templateKey}-draft-${Date.now()}`, version: '', isDefault: false, isSaved: false });
    setDetailPack(null);
  }
  function handleSaveVariation() {
    if (!draftPack || !draftPack.version.trim()) return;
    const saved: TargetPack = { ...draftPack, isSaved: true, id: `${draftPack.templateKey}-${draftPack.version.toLowerCase().replace(/\s+/g, '-')}` };
    const updatedPacks = [...packs, saved];
    setPacks(updatedPacks);
    setDraftPack(null);
    setDetailPack(saved);
    persistPacks(updatedPacks);
  }

  function handleDeleteVariation(packId: string) {
    const updatedPacks = packs.filter((p) => p.id !== packId);
    setPacks(updatedPacks);
    setDetailPack(null);
    persistPacks(updatedPacks);
  }

  function handleRenameVariation(packId: string, newVersion: string) {
    const updatedPacks = packs.map((p) => (p.id === packId ? { ...p, version: newVersion } : p));
    setPacks(updatedPacks);
    setDetailPack((prev) => prev && prev.id === packId ? { ...prev, version: newVersion } : prev);
    persistPacks(updatedPacks);
  }

  if (activePack) {
    return (
      <div>
        <PageHeader
          title={isDraft ? `New Variation of ${activePack.name}` : `${activePack.name} (${activePack.version})`}
          subtitle={isDraft ? 'Configure parameters and preview before saving' : activePack.tags.join(' \u00b7 ')}
          backHref="#"
          actions={
            <div className="flex items-center gap-3">
              {isDraft ? (
                <>
                  <div className="flex items-center gap-2">
                    <label className="text-xs" style={{ color: 'var(--text-muted)' }}>Name:</label>
                    <input className="px-2 py-1.5 rounded-lg text-sm w-40" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} placeholder="e.g. Conservative, 3:1" value={activePack.version} onChange={(e) => setDraftPack({ ...activePack, version: e.target.value })} />
                  </div>
                  <button onClick={handleSaveVariation} className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--accent)', color: 'white', opacity: activePack.version.trim() ? 1 : 0.5, cursor: activePack.version.trim() ? 'pointer' : 'not-allowed' }} disabled={!activePack.version.trim()}>Save as Variation</button>
                  <button onClick={() => setDraftPack(null)} className="px-3 py-1.5 rounded-lg text-xs" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>Discard</button>
                </>
              ) : (
                <>
                  <button onClick={() => handleCopy(activePack)} className="px-3 py-1.5 rounded-lg text-xs" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>Create Variation</button>
                  <button onClick={() => setDetailPack(null)} className="px-4 py-2 rounded-lg text-sm" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>Back to Packs</button>
                </>
              )}
            </div>
          }
        />
        {isDraft && (
          <div className="mb-4 px-4 py-3 rounded-xl flex items-start gap-2" style={{ background: 'var(--orange-muted)', border: '1px solid var(--orange)' }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="flex-shrink-0 mt-0.5" style={{ color: 'var(--orange)' }}><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" /><line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" /></svg>
            <div><p className="text-xs font-medium" style={{ color: 'var(--orange)' }}>Draft mode</p><p className="text-xs" style={{ color: 'var(--text-secondary)' }}>Parameters will be <strong>permanently locked</strong> once saved. Use the Preview tab to test target placement before saving.</p></div>
          </div>
        )}
        <TabBar tabs={['Parameters', 'Behavior', 'Preview', 'Code', 'Danger Zone']}>
          {(tab) => (
            <div>
              {tab === 'Parameters' && (
                <ParametersTab
                  pack={activePack}
                  onParamChange={isDraft ? (key, val) => {
                    setDraftPack({
                      ...activePack,
                      params: activePack.params.map((p) =>
                        p.key === key ? { ...p, value: val } : p),
                    });
                  } : undefined}
                />
              )}
              {tab === 'Behavior' && <BehaviorTab pack={activePack} />}
              {tab === 'Preview' && <PreviewTab pack={activePack} />}
              {tab === 'Code' && <CodeTab pack={activePack} />}
              {tab === 'Danger Zone' && <DangerZoneTab pack={activePack} onRename={(v) => handleRenameVariation(activePack.id, v)} onDelete={() => handleDeleteVariation(activePack.id)} />}
            </div>
          )}
        </TabBar>
      </div>
    );
  }

  return (
    <div>
      <PageHeader title="Take Profit Packs" subtitle="Reusable take profit configurations with locked parameters for consistent backtesting" actions={
        <Link href="/confluence-packs/pack-builder" className="px-4 py-2 rounded-lg text-sm font-medium inline-flex items-center gap-2" style={{ background: 'var(--accent)', color: 'white' }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" /></svg>
          Create New Template
        </Link>
      } />
      <div className="flex items-center gap-4 mb-5">
        <p className="text-sm flex-shrink-0" style={{ color: 'var(--text-muted)' }}>
          {packs.length} packs, {enabledCount} enabled
          {saveStatus === 'saving' && <span className="ml-2 text-xs" style={{ color: 'var(--accent)' }}>Saving...</span>}
          {saveStatus === 'saved' && <span className="ml-2 text-xs" style={{ color: 'var(--green)' }}>Saved</span>}
          {saveStatus === 'error' && <span className="ml-2 text-xs" style={{ color: 'var(--red)' }}>Save failed</span>}
        </p>
        <div className="flex-1 relative">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: 'var(--text-muted)' }}><circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" /></svg>
          <input className="w-full pl-9 pr-3 py-2 rounded-lg text-sm" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} placeholder="Search take profit packs..." value={search} onChange={(e) => setSearch(e.target.value)} />
        </div>
      </div>
      <div className="space-y-2">
        {filteredGroups.map((group) => {
          const hasVariations = group.variations.length > 0;
          const isExpanded = expandedTemplates.has(group.templateKey);
          return (
            <div key={group.templateKey}>
              <PackCard pack={group.default} onToggle={() => togglePack(group.default.id)} onDetails={() => setDetailPack(group.default)} onCopy={() => handleCopy(group.default)} hasVariations={hasVariations} isExpanded={isExpanded} onToggleExpand={() => toggleTemplate(group.templateKey)} />
              {hasVariations && !isExpanded && <button onClick={() => toggleTemplate(group.templateKey)} className="ml-10 mt-1 text-[10px]" style={{ color: 'var(--text-muted)' }}>{group.variations.length} variation{group.variations.length !== 1 ? 's' : ''}</button>}
              {isExpanded && group.variations.map((v) => <div key={v.id} className="mt-2"><PackCard pack={v} onToggle={() => togglePack(v.id)} onDetails={() => setDetailPack(v)} onCopy={() => handleCopy(v)} /></div>)}
            </div>
          );
        })}
        {filteredGroups.length === 0 && <div className="text-center py-12"><p className="text-sm" style={{ color: 'var(--text-muted)' }}>No packs match &quot;{search}&quot;</p></div>}
      </div>
    </div>
  );
}
