'use client';

/**
 * General Packs — Clean API-first page.
 *
 * Visual design adapted from V5 (versions/V5.tsx), data layer built
 * around actual Supabase API response shapes. No mock data, no fallbacks.
 */

import { useState, useMemo } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import { useGeneralPacks, useGeneralTemplates } from '@/hooks/queries/usePacks';
import { useSaveGeneralPacks } from '@/hooks/mutations/usePackMutations';
import type { GeneralPackDTO, GeneralTemplateDTO } from '@/hooks/queries/usePacks';

// ---------------------------------------------------------------------------
// Style constants (from V5)
// ---------------------------------------------------------------------------

const tagColors: Record<string, { color: string; bg: string }> = {
  Time: { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  Session: { color: 'var(--accent)', bg: 'var(--accent-muted)' },
  Calendar: { color: 'var(--orange)', bg: 'var(--orange-muted)' },
  Events: { color: 'var(--red)', bg: 'var(--red-muted)' },
};

const selectStyle: React.CSSProperties = {
  padding: '6px 10px',
  borderRadius: '6px',
  border: '1px solid var(--border)',
  background: 'var(--bg-input)',
  color: 'var(--text-primary)',
  fontSize: '12px',
};

// ---------------------------------------------------------------------------
// Small components
// ---------------------------------------------------------------------------

function TagBadge({ tag }: { tag: string }) {
  const style = tagColors[tag] || { color: 'var(--text-muted)', bg: 'var(--bg-input)' };
  return (
    <span
      className="text-[10px] font-medium px-1.5 py-0.5 rounded"
      style={{ color: style.color, background: style.bg }}
    >
      {tag}
    </span>
  );
}

function Toggle({ enabled, onChange }: { enabled: boolean; onChange: () => void }) {
  return (
    <button
      onClick={(e) => { e.stopPropagation(); onChange(); }}
      className="w-9 h-5 rounded-full relative flex-shrink-0 transition-colors"
      style={{
        background: enabled ? 'var(--accent)' : 'var(--bg-input)',
        border: enabled ? 'none' : '1px solid var(--border)',
      }}
    >
      <div
        className="w-3.5 h-3.5 rounded-full absolute transition-all"
        style={{
          background: enabled ? 'white' : 'var(--text-muted)',
          top: '3px',
          left: enabled ? '19px' : '3px',
        }}
      />
    </button>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a human-readable condition logic summary from parameters. */
function formatConditionLogic(
  templateKey: string,
  parameters: Record<string, unknown>,
  schema: Record<string, { type: string; default: unknown; label: string }> | undefined,
): string {
  if (!schema) return '';

  // Time-of-day: display as "HH:MM-HH:MM ET"
  if (templateKey === 'time_of_day' || (parameters.start_hour != null && parameters.end_hour != null)) {
    const sh = String(parameters.start_hour ?? 9).padStart(2, '0');
    const sm = String(parameters.start_minute ?? 0).padStart(2, '0');
    const eh = String(parameters.end_hour ?? 16).padStart(2, '0');
    const em = String(parameters.end_minute ?? 0).padStart(2, '0');
    return `${sh}:${sm}\u2013${eh}:${em} ET`;
  }

  if (templateKey === 'trading_session' && parameters.session) {
    const labels: Record<string, string> = { pre_market: 'Pre-Market (4:00\u20139:30)', regular: 'Regular (9:30\u201316:00)', after_hours: 'After Hours (16:00\u201320:00)', extended: 'Extended (4:00\u201320:00)' };
    return labels[parameters.session as string] || String(parameters.session);
  }

  // Day of week: show active days
  if (templateKey === 'day_of_week') {
    const days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday'];
    const shortDays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'];
    const active = days.map((d, i) => parameters[d] !== false ? shortDays[i] : null).filter(Boolean);
    return active.join(', ') || 'No days selected';
  }

  // Calendar filter: show active filters
  if (templateKey === 'calendar_filter') {
    const parts: string[] = [];
    if (parameters.avoid_fomc) parts.push('FOMC');
    if (parameters.avoid_nfp) parts.push('NFP');
    if (parameters.avoid_opex) parts.push('OpEx');
    const buf = parameters.buffer_minutes;
    return parts.length > 0
      ? `Avoid: ${parts.join(', ')}${buf ? ` (${buf}min buffer)` : ''}`
      : 'No filters active';
  }

  // Fallback: key=value summary
  return Object.entries(schema)
    .map(([key, s]) => {
      const val = parameters[key] ?? s.default;
      if (s.type === 'bool') return `${s.label}: ${val ? 'Yes' : 'No'}`;
      return `${s.label}: ${val}`;
    })
    .join(', ');
}

function formatParamSummary(params: Record<string, unknown>, schema: Record<string, { type: string; default: unknown; label: string }> | undefined): string {
  if (!schema) return '';
  return Object.entries(schema).map(([k, s]) => { const v = params[k] ?? s.default; return s.type === 'bool' ? `${s.label}: ${v ? 'Y' : 'N'}` : `${s.label}: ${v}`; }).join(', ');
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function GeneralPacksPage() {
  const { data: packs, isLoading: packsLoading, error: packsError } = useGeneralPacks();
  const { data: templates, isLoading: templatesLoading, error: templatesError } = useGeneralTemplates();
  const saveMutation = useSaveGeneralPacks();

  const isLoading = packsLoading || templatesLoading;
  const error = packsError || templatesError;

  // Local state for toggles (optimistic UI)
  const [localToggles, setLocalToggles] = useState<Record<string, boolean>>({});

  // Filter / search state
  const [search, setSearch] = useState('');
  const [categoryFilter, setCategoryFilter] = useState('All');

  // Categories derived from templates
  const allCategories = useMemo(() => {
    if (!templates) return [];
    return Array.from(new Set(Object.values(templates).map((t) => t.category))).sort();
  }, [templates]);

  // Filtered packs
  const filtered = useMemo(() => {
    if (!packs || !templates) return [];
    let result = [...packs];

    if (categoryFilter !== 'All') {
      result = result.filter((p) => {
        const tmpl = templates[p.base_template];
        return tmpl && tmpl.category === categoryFilter;
      });
    }

    if (search.trim()) {
      const q = search.toLowerCase();
      result = result.filter((p) => {
        const tmpl = templates[p.base_template];
        const name = tmpl?.name || p.base_template;
        return (
          name.toLowerCase().includes(q) ||
          p.version.toLowerCase().includes(q) ||
          p.base_template.toLowerCase().includes(q) ||
          (p.description || '').toLowerCase().includes(q)
        );
      });
    }

    return result;
  }, [packs, templates, categoryFilter, search]);

  // Group by base_template
  const groupedPacks = useMemo(() => {
    const map = new Map<string, GeneralPackDTO[]>();
    for (const p of filtered) {
      const key = p.base_template;
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(p);
    }
    Array.from(map.values()).forEach((arr) => {
      arr.sort((a: GeneralPackDTO, b: GeneralPackDTO) => (b.is_default ? 1 : 0) - (a.is_default ? 1 : 0));
    });
    return map;
  }, [filtered]);

  const [expandedTemplates, setExpandedTemplates] = useState<Set<string>>(new Set());

  function handleToggle(pack: GeneralPackDTO) {
    const currentEnabled = localToggles[pack.id] ?? pack.enabled;
    const newEnabled = !currentEnabled;
    setLocalToggles((prev) => ({ ...prev, [pack.id]: newEnabled }));

    if (packs) {
      const updated = packs.map((p) =>
        p.id === pack.id ? { ...p, enabled: newEnabled } : p,
      );
      saveMutation.mutate(updated);
    }
  }

  function isEnabled(pack: GeneralPackDTO): boolean {
    return localToggles[pack.id] ?? pack.enabled;
  }

  // ---------------------------------------------------------------------------
  // Loading state
  // ---------------------------------------------------------------------------

  if (isLoading) {
    return (
      <div>
        <PageHeader title="General Packs" subtitle="Loading..." />
        <div className="space-y-3 mt-4">
          {[1, 2, 3, 4].map((i) => (
            <Card key={i}>
              <div className="animate-pulse flex items-center gap-3">
                <div className="w-9 h-5 rounded-full" style={{ background: 'var(--border)' }} />
                <div className="flex-1 space-y-2">
                  <div className="h-4 rounded w-1/3" style={{ background: 'var(--border)' }} />
                  <div className="h-3 rounded w-2/3" style={{ background: 'var(--border)' }} />
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // Error state
  // ---------------------------------------------------------------------------

  if (error) {
    return (
      <div>
        <PageHeader title="General Packs" subtitle="Error loading data" />
        <Card>
          <div className="text-center py-8" style={{ color: 'var(--red)' }}>
            Failed to load general packs. Check your connection and try again.
          </div>
        </Card>
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  const enabledCount = packs?.filter((p) => isEnabled(p)).length ?? 0;
  const totalCount = packs?.length ?? 0;

  return (
    <div>
      <PageHeader
        title="General Packs"
        subtitle={`${enabledCount} of ${totalCount} enabled`}
        actions={
          <span
            className="text-xs px-2 py-1 rounded-full flex items-center gap-1.5"
            style={{ background: 'var(--green)' + '15', color: 'var(--green)' }}
          >
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--green)' }} />
            Live
          </span>
        }
      />

      {/* Filter row */}
      <div className="flex flex-wrap gap-2 mb-4 mt-4">
        <input
          type="text"
          placeholder="Search packs..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="px-3 py-1.5 rounded-lg text-sm flex-1 min-w-[180px]"
          style={{
            background: 'var(--bg-input)',
            border: '1px solid var(--border)',
            color: 'var(--text-primary)',
          }}
        />
        <select
          style={selectStyle}
          value={categoryFilter}
          onChange={(e) => setCategoryFilter(e.target.value)}
        >
          <option value="All">All Categories</option>
          {allCategories.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
      </div>

      {/* Empty state */}
      {filtered.length === 0 && (
        <Card>
          <div className="text-center py-12">
            <p className="text-lg mb-2" style={{ color: 'var(--text-secondary)' }}>
              {totalCount === 0
                ? 'No general packs configured yet.'
                : 'No packs match the current filters.'}
            </p>
          </div>
        </Card>
      )}

      {/* Pack cards */}
      <div className="space-y-2">
        {Array.from(groupedPacks.entries()).map(([templateKey, packList]) => {
          const tmpl = templates?.[templateKey];
          const defaultPack = packList.find((p) => p.is_default);
          const variations = packList.filter((p) => !p.is_default);
          const hasVariations = variations.length > 0;
          const isExpanded = expandedTemplates.has(templateKey);

          return (
            <div key={templateKey}>
              {defaultPack && (
                <GeneralPackCard
                  pack={defaultPack}
                  template={tmpl}
                  templateKey={templateKey}
                  enabled={isEnabled(defaultPack)}
                  onToggle={() => handleToggle(defaultPack)}
                  hasVariations={hasVariations}
                  variationCount={variations.length}
                  isExpanded={isExpanded}
                  onToggleExpand={() => {
                    setExpandedTemplates((prev) => {
                      const next = new Set(prev);
                      if (next.has(templateKey)) next.delete(templateKey);
                      else next.add(templateKey);
                      return next;
                    });
                  }}
                />
              )}

              {isExpanded && variations.map((v) => (
                <GeneralPackCard
                  key={v.id}
                  pack={v}
                  template={tmpl}
                  templateKey={templateKey}
                  enabled={isEnabled(v)}
                  onToggle={() => handleToggle(v)}
                  isVariation
                />
              ))}
            </div>
          );
        })}
      </div>

      {saveMutation.isPending && (
        <div
          className="fixed bottom-4 right-4 px-4 py-2 rounded-lg text-sm"
          style={{ background: 'var(--accent)', color: 'white' }}
        >
          Saving...
        </div>
      )}
    </div>
  );
}

function GeneralPackCard({
  pack,
  template,
  templateKey,
  enabled,
  onToggle,
  hasVariations,
  variationCount,
  isExpanded,
  onToggleExpand,
  isVariation,
}: {
  pack: GeneralPackDTO;
  template?: GeneralTemplateDTO;
  templateKey: string;
  enabled: boolean;
  onToggle: () => void;
  hasVariations?: boolean;
  variationCount?: number;
  isExpanded?: boolean;
  onToggleExpand?: () => void;
  isVariation?: boolean;
}) {
  const name = template?.name || pack.base_template;
  const category = template?.category || 'Other';
  const conditionSummary = formatConditionLogic(templateKey, pack.parameters, template?.parameters_schema);
  const paramSummary = formatParamSummary(pack.parameters, template?.parameters_schema);
  const outputCount = template?.outputs?.length || 0;
  const triggerCount = template?.triggers?.length || 0;

  return (
    <div
      className="flex items-center gap-3 px-4 py-3 rounded-xl border transition-colors mb-2"
      style={{
        background: 'var(--bg-card)',
        borderColor: 'var(--border)',
        boxShadow: 'var(--card-shadow)',
        backdropFilter: 'var(--card-backdrop)',
        WebkitBackdropFilter: 'var(--card-backdrop)',
        opacity: enabled ? 1 : 0.6,
        marginLeft: isVariation ? 24 : 0,
      }}
    >
      {hasVariations ? (
        <button
          onClick={(e) => { e.stopPropagation(); onToggleExpand?.(); }}
          className="w-5 h-5 rounded flex items-center justify-center text-xs flex-shrink-0 transition-transform"
          style={{ color: 'var(--text-muted)', transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)' }}
        >
          {'\u25B6'}
        </button>
      ) : isVariation ? (
        <div className="w-5 flex-shrink-0" />
      ) : null}

      <Toggle enabled={enabled} onChange={onToggle} />

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1 flex-wrap">
          <span className="font-semibold text-sm" style={{ color: 'var(--text-primary)' }}>
            {name}
          </span>
          <span
            className="text-[10px] px-1.5 py-0.5 rounded"
            style={{
              color: pack.is_default ? 'var(--text-muted)' : 'var(--text-secondary)',
              background: 'var(--bg-input)',
            }}
          >
            {pack.version}
          </span>
          <TagBadge tag={category} />
          {hasVariations && variationCount != null && (
            <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>
              +{variationCount} variation{variationCount !== 1 ? 's' : ''}
            </span>
          )}
        </div>

        {/* Condition logic summary — template-specific display */}
        {conditionSummary && (
          <p className="text-xs font-medium mb-0.5" style={{ color: 'var(--accent)' }}>
            {conditionSummary}
          </p>
        )}

        {/* Raw parameter summary */}
        <p className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>
          {paramSummary}
        </p>
      </div>

      <div className="flex items-center gap-2 flex-shrink-0">
        <span
          className="text-[10px] px-1.5 py-0.5 rounded"
          style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}
        >
          {outputCount} states
        </span>
        {triggerCount > 0 && (
          <span
            className="text-[10px] px-1.5 py-0.5 rounded"
            style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}
          >
            {triggerCount} triggers
          </span>
        )}
      </div>
    </div>
  );
}
